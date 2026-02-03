# test_analysis.py
import numpy as np
import pandas as pd
import pytest

from ncpi import Analysis, tools

# If your package exports PosthocConfig, import it from ncpi.
# If it doesn't, import it from the module where Analysis is defined.
# Adjust this import to match your package layout.
try:
    from ncpi import PosthocConfig  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: if PosthocConfig isn't exported at package level
    # You may need to change this to your actual module path.
    from ncpi.analysis import PosthocConfig  # type: ignore


# -----------------------------
# Backend availability checks
# -----------------------------
def _r_backend_available() -> bool:
    """Return True if rpy2 is importable; False otherwise."""
    return tools.ensure_module("rpy2")


def _mne_available() -> bool:
    """Return True if mne is importable; False otherwise."""
    return tools.ensure_module("mne")


# -----------------------------
# Toy data generation
# -----------------------------
def _generate_toy_data(seed: int = 0) -> pd.DataFrame:
    """Generate a small deterministic dataset for lmer/lm tests."""
    np.random.seed(seed)

    nsubj, nsensors, nepochs = 10, 3, 5
    subj_id = np.repeat(np.arange(nsubj), nsensors * nepochs)
    sensor = np.tile(np.repeat(np.arange(nsensors), nepochs), nsubj)
    epoch = np.tile(np.arange(nepochs), nsensors * nsubj)

    data = pd.DataFrame({"id": subj_id, "epoch": epoch, "sensor": sensor})

    # groups: g1, g2, HC
    group_size = nsubj // 3
    data["gr"] = "HC"
    data.loc[data["id"].isin(range(group_size)), "gr"] = "g1"
    data.loc[data["id"].isin(range(group_size, group_size * 2)), "gr"] = "g2"

    # baseline outcome
    data["Y"] = np.random.normal(size=len(data))

    # Group*Sensor effects
    halfsensor = nsensors // 2
    mask_g1_low_sensor = (data["gr"] == "g1") & (data["sensor"] <= halfsensor)
    data.loc[mask_g1_low_sensor, "Y"] += np.random.normal(
        loc=10, scale=2, size=mask_g1_low_sensor.sum()
    )

    mask_g2 = data["gr"] == "g2"
    data.loc[mask_g2, "Y"] += np.random.normal(loc=7, scale=0.5, size=mask_g2.sum())

    # Make Y vary with epoch for HC and g2 but not for g1
    mask_not_g1 = data["gr"] != "g1"
    data.loc[mask_not_g1, "Y"] += np.random.normal(
        loc=data.loc[mask_not_g1, "epoch"], scale=0.5
    )

    # Subject-level deviations (random intercept-like)
    raneff = pd.DataFrame(
        np.random.normal(loc=0, scale=3, size=nsubj),
        index=range(nsubj),
        columns=["raneff"],
    )
    data = data.join(raneff, on="id")
    data["Y"] = data["Y"] + np.random.normal(loc=data["raneff"], scale=0.1)
    data = data.drop(columns="raneff")

    # Covariate
    data["sex"] = np.random.choice(["F", "M"], size=len(data))

    return data


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
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr * epoch * sensor + sex + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )
    assert isinstance(opt_f, str)
    assert "~" in opt_f
    # normalize spaces not assumed; we just ensure dependent variable is Y
    assert opt_f.replace(" ", "").startswith("Y~")


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
def test_lmer_tests_skips_invalid_specs(analysis: Analysis):
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr * epoch * sensor + sex + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )

    results = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["epoch:gr", "gr|sensor", "x"],  # "x" does not exist -> should be skipped
        print_info=False,
    )

    assert isinstance(results, dict)
    assert "x" not in results
    assert any(k in results for k in ["epoch:gr", "gr|sensor"])


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_defaults_specs_to_group_col(analysis: Analysis):
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr * epoch * sensor + sex + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )

    results = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=None,
        print_info=False,
    )

    assert "gr" in results
    assert isinstance(results["gr"], pd.DataFrame)


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_lmer_tests_bic_selects_best_from_given_models(analysis: Analysis):
    results = analysis.lmer_tests(
        models=[
            "Y ~ gr * epoch * sensor + sex + (1|sensor) + (1|id)",
            "Y ~ gr * epoch * sensor + (1|sex) + (1|id)",
        ],
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr|sensor", "epoch:gr"],
        print_info=False,
    )

    assert isinstance(results, dict)
    assert any(k in results for k in ["gr|sensor", "epoch:gr"])


# -----------------------------
# Posthoc configurability tests
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
@pytest.mark.parametrize("adjust", ["holm", "bonferroni", "fdr", "tukey"])
def test_posthoc_adjustments_run(analysis: Analysis, adjust: str):
    """Ensure different p-value adjustments don't crash and return a DataFrame."""
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr * epoch * sensor + sex + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )

    results = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["gr"],
        posthoc={"adjust": adjust},
        print_info=False,
    )
    assert "gr" in results
    assert isinstance(results["gr"], pd.DataFrame)


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_pairs_vs_contrast_methods(analysis: Analysis):
    """Exercise both pairwise implementations for non-control comparisons."""
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr * epoch * sensor + sex + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )

    # All-pairs via pairs()
    res_pairs = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group=None,  # ensures all-pairs path
        numeric=["epoch"],
        specs=["gr"],
        posthoc=PosthocConfig(pairwise_method="pairs"),
        print_info=False,
    )
    assert "gr" in res_pairs

    # All-pairs via contrast(method="pairwise")
    res_contrast = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group=None,
        numeric=["epoch"],
        specs=["gr"],
        posthoc={"pairwise_method": "contrast", "pairwise_contrast_method": "pairwise"},
        print_info=False,
    )
    assert "gr" in res_contrast


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_invalid_pairwise_method_raises(analysis: Analysis):
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )
    with pytest.raises(ValueError, match="pairwise_method"):
        analysis.lmer_tests(
            models=opt_f,
            group_col="gr",
            control_group=None,
            specs=["gr"],
            posthoc={"pairwise_method": "nope"},
            print_info=False,
        )


@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_posthoc_treatment_ref_override(analysis: Analysis):
    """Verify treatment_ref can override control_group as reference."""
    opt_f = analysis.lmer_selection(
        full_model="Y ~ gr + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )

    # Here we pass control_group="HC" but override reference to "g1".
    # This mainly checks the pathway works without crashing.
    results = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group="HC",
        specs=["gr"],
        posthoc={"treatment_ref": "g1"},
        print_info=False,
    )
    assert "gr" in results
    assert isinstance(results["gr"], pd.DataFrame)


# -----------------------------
# Trend (emtrends) behavior tests
# -----------------------------
@pytest.mark.skipif(not _r_backend_available(), reason="rpy2 (and R) not available")
def test_emtrends_numeric_spec_runs(analysis: Analysis):
    """Spec containing a numeric variable should run emtrends/test path."""
    opt_f = analysis.lmer_selection(
        full_model="Y ~ epoch * gr + (1|id)",
        numeric=["epoch"],
        random_crit="BIC",
        fixed_crit=None,
        print_info=False,
    )
    results = analysis.lmer_tests(
        models=opt_f,
        group_col="gr",
        control_group="HC",
        numeric=["epoch"],
        specs=["epoch:gr"],  # slope per group
        print_info=False,
    )
    assert "epoch:gr" in results
    assert isinstance(results["epoch:gr"], pd.DataFrame)


# -----------------------------
# cohend tests
# -----------------------------
def test_cohend_outputs_expected_keys(toy_df: pd.DataFrame):
    """cohend should compute per-sensor d for each non-control group vs control."""
    # cohend defaults assume group_col="group" but our data uses "gr"
    a = Analysis(toy_df)
    res = a.cohend(control_group="HC", group_col="gr", sensor_col="sensor", data_col="Y", min_n=2)
    assert isinstance(res, dict)
    assert set(res.keys()) == {"g1vsHC", "g2vsHC"}
    for k, df in res.items():
        assert isinstance(df, pd.DataFrame)
        assert "sensor" in df.columns
        assert "d" in df.columns


# -----------------------------
# EEG topomap tests (MNE)
# -----------------------------
@pytest.mark.skipif(not _mne_available(), reason="mne not available")
def test_eeg_topomap_accepts_dict_and_series():
    """eeg_topomap should accept dict and pandas Series channel-value inputs."""
    a = Analysis(pd.DataFrame({"x": [1]}))  # data irrelevant for eeg_topomap

    vals_dict = {"Fp1": 1.0, "Fp2": 0.5, "Fz": -0.1, "Cz": 0.2}
    im, cn = a.eeg_topomap(vals_dict, montage="standard_1020", show=False, colorbar=False)
    assert im is not None

    vals_series = pd.Series(vals_dict)
    im2, cn2 = a.eeg_topomap(vals_series, montage="standard_1020", show=False, colorbar=False)
    assert im2 is not None


@pytest.mark.skipif(not _mne_available(), reason="mne not available")
def test_eeg_topomap_requires_ch_names_for_array():
    a = Analysis(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="must provide `ch_names`"):
        a.eeg_topomap(np.array([1.0, 2.0, 3.0]), montage="standard_1020", show=False)


@pytest.mark.skipif(not _mne_available(), reason="mne not available")
def test_eeg_topomap_length_mismatch_raises():
    a = Analysis(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="Length mismatch"):
        a.eeg_topomap(np.array([1.0, 2.0, 3.0]), ch_names=["Fp1", "Fp2"], montage="standard_1020", show=False)
