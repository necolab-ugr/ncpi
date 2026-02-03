# test_analysis.py
import numpy as np
import pandas as pd
import pytest

from ncpi import Analysis, tools

# If your package exports PosthocConfig, import it from ncpi.
# If it doesn't, import it from the module where Analysis is defined.
try:
    from ncpi import PosthocConfig  # type: ignore
except Exception:  # pragma: no cover
    from ncpi.analysis import PosthocConfig  # type: ignore

# If your package exports LmerTestsResult, import it for type checks
try:
    from ncpi import LmerTestsResult  # type: ignore
except Exception:  # pragma: no cover
    try:
        from ncpi.analysis import LmerTestsResult  # type: ignore
    except Exception:  # pragma: no cover
        LmerTestsResult = None  # type: ignore


# -----------------------------
# Backend availability checks
# -----------------------------
def _r_backend_available() -> bool:
    """Return True if rpy2 is importable; False otherwise."""
    return tools.ensure_module("rpy2")


# -----------------------------
# Toy data generation
# -----------------------------
def _generate_toy_data(seed: int = 0) -> pd.DataFrame:
    """Generate a small deterministic dataset for lm/lmer/emmeans tests."""
    rng = np.random.default_rng(seed)

    nsubj, nsensors, nepochs = 12, 3, 5
    subj_id = np.repeat(np.arange(nsubj), nsensors * nepochs)
    sensor = np.tile(np.repeat(np.arange(nsensors), nepochs), nsubj)
    epoch = np.tile(np.arange(nepochs), nsensors * nsubj)

    df = pd.DataFrame({"id": subj_id, "epoch": epoch, "sensor": sensor})

    # 3 groups: HC, g1, g2 (balanced-ish)
    df["gr"] = "HC"
    df.loc[df["id"].isin(range(0, 4)), "gr"] = "g1"
    df.loc[df["id"].isin(range(4, 8)), "gr"] = "g2"

    df["sex"] = rng.choice(["F", "M"], size=len(df), replace=True)

    # Random intercept-like subject effect
    subj_re = rng.normal(loc=0.0, scale=2.0, size=nsubj)
    df["subj_re"] = df["id"].map({i: subj_re[i] for i in range(nsubj)})

    # Baseline signal
    y = rng.normal(loc=0.0, scale=1.0, size=len(df))

    # Add group main effects
    y += (df["gr"] == "g1") * 4.0
    y += (df["gr"] == "g2") * 2.5

    # Add epoch slope for HC and g2, but not g1
    y += ((df["gr"] != "g1").astype(float)) * (0.8 * df["epoch"].astype(float))

    # Add an interaction-ish sensor effect for g1 on sensor 0
    y += ((df["gr"] == "g1") & (df["sensor"] == 0)).astype(float) * 3.0

    # Add subject random effect
    y += df["subj_re"].to_numpy()

    # Small noise
    y += rng.normal(loc=0.0, scale=0.2, size=len(df))

    df["Y"] = y
    df = df.drop(columns=["subj_re"])

    # Ensure sensor is categorical in R unless included as numeric; keep as int in pandas
    return df


@pytest.fixture(scope="module")
def toy_df() -> pd.DataFrame:
    return _generate_toy_data(seed=0)


@pytest.fixture(scope="module")
def analysis(toy_df: pd.DataFrame) -> Analysis:
    return Analysis(toy_df)


# -----------------------------
# Core sanity / validation tests
# -----------------------------
def test_requires_dataframe_for_stats_methods():
    """lmer_tests/lmer_selection/cohend should fail if Analysis.data is not a DataFrame."""
    a = Analysis(data=[1, 2, 3])
    with pytest.raises(TypeError):
        a.lmer_selection(full_model="Y ~ x", numeric=["x"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        a.lmer_tests(models="Y ~ x", group_col="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        a.cohend()  # type: ignore[arg-type]


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_selection_returns_formula(analysis: Analysis):
    f = analysis.lmer_selection(
        full_model="Y ~ gr * epoch * sensor + sex + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )
    assert isinstance(f, str)
    assert "~" in f
    assert f.replace(" ", "").startswith("Y~")


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_selection_rejects_invalid_formula(analysis: Analysis):
    with pytest.raises(ValueError, match="missing"):
        analysis.lmer_selection(full_model="Y gr + (1|id)", numeric=["epoch"], print_info=False)


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_rejects_inconsistent_dependent_variable(analysis: Analysis):
    with pytest.raises(ValueError, match="same dependent variable"):
        analysis.lmer_tests(
            models=["Y ~ gr + (1|id)", "Z ~ gr + (1|id)"],
            group_col="gr",
            print_info=False,
        )


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_requires_group_col_when_specs_none(analysis: Analysis):
    f = "Y ~ gr + (1|id)"
    with pytest.raises(ValueError, match="group_col must be provided"):
        analysis.lmer_tests(models=f, group_col=None, specs=None, print_info=False)


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_invalid_control_group_rejected(analysis: Analysis):
    f = "Y ~ gr + (1|id)"
    with pytest.raises(ValueError, match="control_group"):
        analysis.lmer_tests(models=f, group_col="gr", control_group="NOT_A_LEVEL", specs=["gr"], print_info=False)


# -----------------------------
# lmer_tests: selection + spec behavior
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_defaults_specs_to_group_col(analysis: Analysis):
    f = "Y ~ gr * epoch + sex + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=None,
        print_info=False,
    )
    assert isinstance(res, dict)
    assert "gr" in res
    assert isinstance(res["gr"], pd.DataFrame)
    assert len(res["gr"]) > 0


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_skips_specs_with_unknown_vars(analysis: Analysis):
    f = "Y ~ gr * epoch + sex + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr", "epoch:gr", "does_not_exist"],
        print_info=False,
    )
    assert "gr" in res
    assert "epoch:gr" in res
    assert "does_not_exist" not in res


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_skips_specs_with_two_numeric_vars(analysis: Analysis):
    # epoch numeric; create a second numeric covariate
    df = analysis.data.copy()
    df["age"] = np.linspace(20, 60, len(df))
    a = Analysis(df)

    f = "Y ~ gr * epoch + age + (1|id)"
    res = a.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch", "age"],
        specs=["epoch:age:gr"],  # > 1 numeric variable in spec => should be skipped
        print_info=False,
    )
    assert isinstance(res, dict)
    assert "epoch:age:gr" not in res


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_bic_selects_best_from_models_does_not_crash(analysis: Analysis):
    res = analysis.lmer_tests(
        models=[
            "Y ~ gr * epoch * sensor + sex + (1|sensor) + (1|id)",
            "Y ~ gr * epoch * sensor + sex + (1|id)",
        ],
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr|sensor", "epoch:gr"],
        print_info=False,
    )
    assert isinstance(res, dict)
    assert any(k in res for k in ["gr|sensor", "epoch:gr"])


# -----------------------------
# PATCH regression test:
# interaction-only spec should not be skipped
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_interaction_only_model_allows_factor_specs(analysis: Analysis):
    """
    Regression test for the spec-filtering fix:
    If the model has only an interaction term (e.g., gr:sensor) and not main effects,
    specs like 'sensor|gr' should still be allowed.
    """
    f = "Y ~ gr:sensor + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],  # epoch is in df but not in model; harmless
        specs=["sensor|gr"],
        print_info=False,
    )
    assert "sensor|gr" in res
    assert isinstance(res["sensor|gr"], pd.DataFrame)
    assert len(res["sensor|gr"]) > 0


# -----------------------------
# Posthoc configurability tests
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
@pytest.mark.parametrize("adjust", ["holm", "bonferroni", "fdr", "tukey"])
def test_posthoc_adjustments_run(analysis: Analysis, adjust: str):
    f = "Y ~ gr * epoch + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr"],
        posthoc={"adjust": adjust},
        print_info=False,
    )
    assert "gr" in res
    assert isinstance(res["gr"], pd.DataFrame)


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_pairs_vs_contrast_methods(analysis: Analysis):
    f = "Y ~ gr + (1|id)"

    res_pairs = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group=None,
        specs=["gr"],
        posthoc=PosthocConfig(pairwise_method="pairs"),
        print_info=False,
    )
    assert "gr" in res_pairs

    res_contrast = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group=None,
        specs=["gr"],
        posthoc={"pairwise_method": "contrast", "pairwise_contrast_method": "pairwise"},
        print_info=False,
    )
    assert "gr" in res_contrast


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_invalid_pairwise_method_raises(analysis: Analysis):
    f = "Y ~ gr + (1|id)"
    with pytest.raises(ValueError, match="pairwise_method"):
        analysis.lmer_tests(
            models=f,
            group_col="gr",
            control_group=None,
            specs=["gr"],
            posthoc={"pairwise_method": "nope"},
            print_info=False,
        )


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_unknown_keys_ignored(analysis: Analysis):
    """
    Ensure unknown keys in posthoc dict don't crash (they should be ignored by coercion).
    """
    f = "Y ~ gr + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group=None,
        specs=["gr"],
        posthoc={"adjust": "holm", "this_key_does_not_exist": 123},
        print_info=False,
    )
    assert "gr" in res


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_treatment_ref_override(analysis: Analysis):
    """
    Verify treatment_ref can override control_group as reference.
    Mainly checks pathway works without crashing.
    """
    f = "Y ~ gr + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        specs=["gr"],
        posthoc={"treatment_ref": "g1"},
        print_info=False,
    )
    assert "gr" in res
    assert isinstance(res["gr"], pd.DataFrame)


# -----------------------------
# Trend (emtrends) behavior tests
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_emtrends_numeric_spec_runs(analysis: Analysis):
    f = "Y ~ epoch * gr + (1|id)"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["epoch:gr"],
        print_info=False,
    )
    assert "epoch:gr" in res
    assert isinstance(res["epoch:gr"], pd.DataFrame)
    assert len(res["epoch:gr"]) > 0


# -----------------------------
# LRT + model_info return types
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lrt_requires_lrt_drop(analysis: Analysis):
    f = "Y ~ gr * epoch + (1|id)"
    with pytest.raises(ValueError, match="requires lrt_drop"):
        analysis.lmer_tests(
            models=f,
            group_col="gr",
            control_group="HC",
            numeric=["epoch"],
            specs=["gr"],
            lrt=True,
            lrt_drop=None,
            print_info=False,
        )


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lrt_returns_result_object_with_lrt(analysis: Analysis):
    f = "Y ~ gr * epoch + (1|id)"
    out = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr"],
        lrt=True,
        lrt_drop=["gr:epoch"],
        return_model_info=True,
        print_info=False,
    )

    # Should be LmerTestsResult-like
    assert hasattr(out, "posthoc")
    assert hasattr(out, "lrt")
    assert hasattr(out, "model_info")

    assert isinstance(out.posthoc, dict)
    assert "gr" in out.posthoc
    assert isinstance(out.posthoc["gr"], pd.DataFrame)

    assert out.lrt is not None
    assert isinstance(out.lrt, pd.DataFrame)
    assert len(out.lrt) > 0

    assert out.model_info is not None
    assert "bics" in out.model_info
    assert "selected_formula" in out.model_info


# -----------------------------
# lm (non-mixed) path sanity
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lm_path_works_with_emmeans(analysis: Analysis):
    """
    If model formula has no random effects, code uses lm().
    Ensure posthoc still runs.
    """
    f = "Y ~ gr * epoch + sex"
    res = analysis.lmer_tests(
        models=f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr"],
        print_info=False,
    )
    assert "gr" in res
    assert isinstance(res["gr"], pd.DataFrame)


# -----------------------------
# cohend tests
# -----------------------------
def test_cohend_outputs_expected_keys(toy_df: pd.DataFrame):
    """cohend should compute per-sensor d for each non-control group vs control."""
    a = Analysis(toy_df)
    res = a.cohend(
        control_group="HC",
        group_col="gr",
        sensor_col="sensor",
        data_col="Y",
        min_n=2,
        drop_zeros=False,
    )
    assert isinstance(res, dict)
    assert set(res.keys()) == {"g1vsHC", "g2vsHC"}
    for k, df in res.items():
        assert isinstance(df, pd.DataFrame)
        assert "sensor" in df.columns
        assert "d" in df.columns
        # should have one row per sensor
        assert df["sensor"].nunique() == toy_df["sensor"].nunique()
