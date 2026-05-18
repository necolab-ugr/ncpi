from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from ncpi import tools


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def extract_variables(formula: str) -> set[str]:
    """Extract variable names from an R-style formula string.

    Parameters
    ----------
    formula:
        An R-style formula string, e.g. ``"Y ~ group * epoch + (1|id)"``.

    Returns
    -------
    set[str]
        Unique token strings that look like variable names.

    Implementation notes
    --------------------
    - We split on common formula operators: ``~ + * : / - | ( )`` and whitespace.
    - We drop tokens that look like numeric constants (e.g., ``1`` or ``0.5``).

    Limitations
    -----------
    - This is a pragmatic tokenizer, not a full R parser.
    - It will treat function names (e.g. ``log(x)`` -> ``log``) as a "variable"
      unless you avoid functions in formulas or filter them out.
    """
    tokens = re.split(r"[\s\+\~\|\(\)\*\/\-\:]+", formula)
    return set(t for t in tokens if t and not t.replace(".", "", 1).isdigit())


@dataclass(frozen=True)
class _RContext:
    """Small container for rpy2 objects we use repeatedly."""

    pandas2ri: Any
    r: Any
    ro: Any


@dataclass(frozen=True)
class PosthocConfig:
    """Configuration for post-hoc testing in :meth:`Analysis.lmer_tests`.

    The defaults reproduce:
    - Holm correction
    - `pairs()` for all-pairs comparisons
    - `trt.vs.ctrl` for treatment-vs-control contrasts
    - reference level = `control_group` (unless overridden)

    Parameters
    ----------
    adjust:
        P-value adjustment method passed to emmeans/contrast/test.
        Common options include: ``'holm'``, ``'tukey'``, ``'bonferroni'``, ``'fdr'``, ``'none'``.
    pairwise_method:
        How to compute all-pairs comparisons when not doing treatment-vs-control:

        - ``'pairs'``: uses ``pairs(emm, adjust=...)`` (default)
        - ``'contrast'``: uses ``contrast(emm, method=pairwise_contrast_method, adjust=...)``
    pairwise_contrast_method:
        Only used if ``pairwise_method == 'contrast'``.
        Common options: ``'pairwise'``, ``'revpairwise'``.
    treatment_method:
        Contrast method for treatment-vs-control when `control_group` is provided
        and the spec corresponds to the primary group_col.
    treatment_ref:
        Override the reference level passed to `contrast(..., ref=...)`.
        If None, uses the Python argument `control_group`.
    """

    adjust: str = "holm"
    pairwise_method: str = "pairs"
    pairwise_contrast_method: str = "pairwise"
    treatment_method: str = "trt.vs.ctrl"
    treatment_ref: Optional[Union[str, int]] = None


@dataclass(frozen=True)
class LmerTestsResult:
    """Return type for :meth:`Analysis.lmer_tests` when extra outputs are requested.

    Attributes
    ----------
    posthoc:
        Dict of pandas DataFrames keyed by spec.
    lrt:
        Likelihood-ratio test table (R ``anova(reduced, full)``), if requested.
    model_info:
        Optional model-selection metadata such as BICs and the selected formula.
    """

    posthoc: Dict[str, pd.DataFrame]
    lrt: Optional[pd.DataFrame] = None
    model_info: Optional[Dict[str, object]] = None


def _coerce_posthoc_config(posthoc: Optional[Union[PosthocConfig, Mapping[str, object]]]) -> PosthocConfig:
    """Coerce user config to PosthocConfig with sane defaults."""
    if posthoc is None:
        return PosthocConfig()
    if isinstance(posthoc, PosthocConfig):
        return posthoc
    allowed = set(PosthocConfig.__dataclass_fields__.keys())
    kwargs = {k: v for k, v in dict(posthoc).items() if k in allowed}
    return PosthocConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------
class Analysis:
    """Statistical analysis and EEG/MEG visualization helpers.

    Parameters
    ----------
    data:
        Primary data container.

        - For statistical methods (`lmer_tests`, `lmer_selection`, `cohend`) this must be a
          :class:`pandas.DataFrame`.
        - EEG plotting uses explicit `values` + channel info; it does not require `self.data`
          to be a particular shape.

    Notes
    -----
    The object stores `data` verbatim. Methods that require a DataFrame call `_as_dataframe()`
    which raises an informative error if `data` is not a DataFrame.
    """

    def __init__(self, data: Any):
        self.data = data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _as_dataframe(self) -> pd.DataFrame:
        """Return `self.data` as a DataFrame or raise a clear error."""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("This method requires Analysis.data to be a pandas DataFrame.")
        return self.data

    @staticmethod
    def _normalize_str_list(x: Union[str, Sequence[str]]) -> List[str]:
        """Normalize a string or sequence of strings into a whitespace-free list."""
        if isinstance(x, str):
            x = [x]
        return [s.replace(" ", "") for s in x]

    @staticmethod
    def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
        """Dedupe while preserving order."""
        seen: set[str] = set()
        out: List[str] = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    def _get_r_context(self, packages: Sequence[str]) -> _RContext:
        """Import rpy2 and ensure required R packages are available."""
        if not tools.ensure_module("rpy2"):
            raise ImportError("rpy2 is required but is not installed.")

        pandas2ri = tools.dynamic_import("rpy2.robjects.pandas2ri")
        r = tools.dynamic_import("rpy2.robjects", "r")
        ro = tools.dynamic_import("rpy2", "robjects")

        pandas2ri.activate()

        pkgs_r = ", ".join([f'"{p}"' for p in packages])
        ro.r(
            f"""
            load_packages <- function(packages) {{
                for (pkg in packages) {{
                    if (!require(pkg, character.only = TRUE)) {{
                        stop(paste0("R package '", pkg, "' is not installed."))
                    }}
                }}
            }}
            load_packages(c({pkgs_r}))
            """
        )
        return _RContext(pandas2ri=pandas2ri, r=r, ro=ro)

    # ------------------------------------------------------------------
    # Mixed model: tests + post-hocs
    # ------------------------------------------------------------------
    def lmer_tests(
        self,
        models: Union[str, Sequence[str]],
        group_col: Optional[str] = None,
        control_group: Optional[str] = None,
        numeric: Optional[Sequence[str]] = None,
        specs: Optional[Union[str, Sequence[str]]] = None,
        *,
        posthoc: Optional[Union[PosthocConfig, Mapping[str, object]]] = None,
        print_info: bool = True,
        lrt: bool = False,
        lrt_drop: Optional[Sequence[str]] = None,
        return_model_info: bool = False,
    ) -> Union[Dict[str, pd.DataFrame], LmerTestsResult]:
        """Fit lm/lmer model(s) in R, optionally select by BIC, then run post-hoc tests.

        Parameters
        ----------
        models:
            One or more R-style model formulas as strings.
            If multiple formulas are provided, the best model is selected by BIC.
        group_col:
            Name of the grouping column (used for treatment-vs-control contrasts).
            Required if `specs` is None.
        control_group:
            Name of the control group level within `group_col` (used for treatment-vs-control
            contrasts). If None, all-pairs comparisons are used.
        numeric:
            List of variable names that should be treated as numeric.
            The dependent variable is always treated as numeric.
        specs:
            One or more spec strings indicating which post-hoc tests to run.
            Each spec is either:
            - A factor variable name (e.g. ``"group"``) for which all-pairs or treatment-vs-control
              comparisons are performed.
            - A numeric variable name (e.g. ``"epoch"``) for which slope
                comparisons are performed.
            If None, defaults to `[group_col]`.
        posthoc:
            PosthocConfig or dict of options to configure post-hoc testing behavior.
        print_info:
            If True, print progress and results to stdout.
        lrt:
            If True, perform a likelihood-ratio test (LRT) between the selected model
            and a reduced model with terms in `lrt_drop` removed.
        lrt_drop:
            List of term labels (as strings) to drop from the selected model
            when constructing the reduced model for LRT.
            Required if `lrt` is True.
        return_model_info:
            If True, return extra model-selection metadata such as BICs
            and the selected formula.

        Returns
        -------
        Union[Dict[str, pd.DataFrame], LmerTestsResult]
            If `lrt` or `return_model_info` is True, returns an LmerTests
            object containing post-hoc results and extra info.
            Otherwise, returns a dict of post-hoc result DataFrames keyed by spec.
        """

        df_in = self._as_dataframe()
        ctx = self._get_r_context(["lme4", "emmeans"])
        pandas2ri, r, ro = ctx.pandas2ri, ctx.r, ctx.ro
        ph_cfg = _coerce_posthoc_config(posthoc)

        # Normalize inputs
        models_list = self._normalize_str_list(models)
        numeric_list = self._dedupe_preserve_order(list(numeric) if numeric is not None else [])

        # Validate formulas and dependent variable consistency
        if "~" not in models_list[0]:
            raise ValueError(f'Invalid model formula "{models_list[0]}": missing "~".')
        y = models_list[0].split("~", 1)[0]
        for m in models_list[1:]:
            if "~" not in m:
                raise ValueError(f'Invalid model formula "{m}": missing "~".')
            if m.split("~", 1)[0] != y:
                raise ValueError("All models must have the same dependent variable (left of ~).")

        # Force dependent variable to numeric
        if y not in numeric_list:
            numeric_list.append(y)

        # Collect variables referenced across models
        all_vars: set[str] = set()
        for m in models_list:
            all_vars |= extract_variables(m)

        # Validate column existence
        missing = [v for v in all_vars if v not in df_in.columns]
        if missing:
            raise ValueError(f"Variables not found in DataFrame columns: {missing}")

        # Validate grouping logic
        if specs is None and group_col is None:
            raise ValueError("If specs is None, group_col must be provided.")
        if group_col is not None:
            if group_col not in all_vars:
                raise ValueError(f'group_col "{group_col}" is not used in the provided model formula(s).')
            if control_group is not None and control_group not in set(df_in[group_col].unique()):
                raise ValueError(f'control_group "{control_group}" not found among values of "{group_col}".')

        # Prepare data for R
        df = df_in.loc[:, sorted(all_vars)].copy()
        ro.globalenv["df"] = pandas2ri.py2rpy(df)
        if control_group is not None:
            ro.globalenv["control_group"] = control_group

        # Convert non-numeric vars to factors (essential for emmeans behavior)
        factor_vars = sorted(set(all_vars) - set(numeric_list))
        if factor_vars:
            r("\n".join([f"df${col} <- as.factor(df${col})" for col in factor_vars]))

        # ------------------------------------------------------------------
        # Fit mixed models with ML (REML=FALSE) for comparisons
        # This makes BIC comparisons and any fixed-effect comparisons canonical.
        # ------------------------------------------------------------------
        for i, formula in enumerate(models_list):
            fit_fun = "lmer" if "|" in formula else "lm"
            if fit_fun == "lmer":
                r(f"m{i} <- lmer({formula}, data=df, REML=FALSE)")
            else:
                r(f"m{i} <- lm({formula}, data=df)")

        # BIC-based selection if more than one candidate
        if len(models_list) == 1:
            r(
                """
                final_model <- m0
                fitted_models <- c("m0")
                bics <- c(BIC(m0))
                names(bics) <- fitted_models
                selected_name <- "m0"
                """
            )
        else:
            ro.globalenv["fitted_models"] = [f"m{i}" for i in range(len(models_list))]
            r(
                """
                fitted_models <- unlist(fitted_models)
                bics <- sapply(fitted_models, function(m) BIC(get(m)))
                index <- which.min(bics)
                selected_name <- fitted_models[index]
                final_model <- get(selected_name)
                """
            )
            if print_info:
                print("--- BIC model selection")

        if print_info:
            print(f"Model: {r('formula(final_model)')}")
            print(
                f"Posthoc: adjust={ph_cfg.adjust}, pairwise_method={ph_cfg.pairwise_method}, "
                f"treatment_method={ph_cfg.treatment_method}"
            )

        lrt_df_pd: Optional[pd.DataFrame] = None
        if lrt:
            if not lrt_drop:
                raise ValueError("lrt=True requires lrt_drop, e.g. ['group:sensor'].")
            ro.globalenv["lrt_drop"] = list(lrt_drop)
            r(
                """
                reduced_model <- final_model
                for (term in lrt_drop) {
                    reduced_model <- update(reduced_model, as.formula(paste(". ~ . -", term)))
                }
                lrt_tbl <- anova(reduced_model, final_model)
                lrt_df <- as.data.frame(lrt_tbl)
                """
            )

            lrt_obj = ro.r["lrt_df"]
            if isinstance(lrt_obj, pd.DataFrame):
                lrt_df_pd = lrt_obj.copy()
            else:
                from rpy2.robjects.conversion import localconverter

                with localconverter(ro.default_converter + pandas2ri.converter):
                    lrt_df_pd = ro.conversion.rpy2py(lrt_obj)

            # Attach a small amount of metadata for logging/debugging
            try:
                selected_name_py = str(r("selected_name")[0])
            except Exception:
                selected_name_py = None
            try:
                selected_formula_py = str(r("deparse(formula(final_model))")[0])
            except Exception:
                selected_formula_py = None

            if selected_name_py is not None:
                lrt_df_pd["selected_model"] = selected_name_py
            if selected_formula_py is not None:
                lrt_df_pd["selected_formula"] = selected_formula_py
            lrt_df_pd["lrt_drop"] = ", ".join(lrt_drop)

        model_info: Optional[Dict[str, object]] = None
        if return_model_info:
            r("bic_df <- data.frame(model=names(bics), bic=as.numeric(bics))")
            bic_obj = ro.r["bic_df"]
            if isinstance(bic_obj, pd.DataFrame):
                bic_df_pd = bic_obj.copy()
            else:
                from rpy2.robjects.conversion import localconverter

                with localconverter(ro.default_converter + pandas2ri.converter):
                    bic_df_pd = ro.conversion.rpy2py(bic_obj)

            try:
                selected_name_py = str(r("selected_name")[0])
            except Exception:
                selected_name_py = None
            try:
                selected_formula_py = str(r("deparse(formula(final_model))")[0])
            except Exception:
                selected_formula_py = None

            model_info = {
                "bics": bic_df_pd,
                "selected_model": selected_name_py,
                "selected_formula": selected_formula_py,
            }

        # ------------------------------------------------------------------
        # Use model variables (not term.labels) to validate specs
        # This avoids incorrectly skipping specs when a variable appears only in interactions.
        # ------------------------------------------------------------------
        selmod_vars = set(r("all.vars(nobars(formula(final_model)))"))
        # Remove response (LHS) if present
        selmod_vars.discard(y)

        # Normalize specs (default to group_col)
        if specs is None:
            specs_list = [group_col]  # type: ignore[list-item]
        else:
            specs_list = self._normalize_str_list(specs)

        SpecInstruction = Union[str, Tuple[str, str]]  # "~spec" OR ("~spec_without_numeric", "numeric_var")
        spec_instructions: List[Tuple[str, SpecInstruction]] = []

        numeric_set = set(numeric_list)
        for sp in specs_list:
            vars_in_sp = extract_variables(sp)

            if any(v not in selmod_vars for v in vars_in_sp):
                if print_info:
                    print(f'(!) Specs "{sp}" skipped: variable not present in selected model formula.')
                continue

            numeric_in_sp = [v for v in vars_in_sp if v in numeric_set]

            if len(numeric_in_sp) == 0:
                spec_instructions.append((sp, "~" + sp))
            elif len(numeric_in_sp) == 1:
                num_var = numeric_in_sp[0]
                if len(vars_in_sp) == 1:
                    spec_formula = "~1"
                else:
                    others = sorted([v for v in vars_in_sp if v != num_var])
                    spec_formula = "~" + ":".join(others)
                spec_instructions.append((sp, (spec_formula, num_var)))
            else:
                if print_info:
                    print(f'(!) Specs "{sp}" skipped: max 1 numeric variable per test.')

        if print_info:
            print("--- Post-hoc tests:")

        # R env config
        ro.globalenv["p_adjust"] = ph_cfg.adjust
        ro.globalenv["pairwise_method"] = ph_cfg.pairwise_method
        ro.globalenv["pairwise_contrast_method"] = ph_cfg.pairwise_contrast_method
        ro.globalenv["trt_method"] = ph_cfg.treatment_method

        results: Dict[str, pd.DataFrame] = {}

        for sp, instr in spec_instructions:
            if isinstance(instr, str):
                # Factor-based comparisons
                ro.globalenv["specs"] = instr
                use_trt_vs_ctrl = (
                    group_col is not None
                    and control_group is not None
                    and instr.split("|", 1)[0] == "~" + group_col
                )

                if use_trt_vs_ctrl:
                    ref_level = control_group if ph_cfg.treatment_ref is None else ph_cfg.treatment_ref
                    ro.globalenv["ref_level"] = ref_level
                    ro.globalenv["group_col_name"] = group_col
                    r(
                        """
                        emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                        ref_idx <- which(levels(df[[group_col_name]]) == ref_level)
                        if (length(ref_idx) != 1) {
                          stop("Control group level not found or not unique in factor levels.")
                        }
                        res <- contrast(emm, method=trt_method, ref=ref_idx, adjust=p_adjust)
                        df_res <- as.data.frame(res)
                        """
                    )
                else:
                    if ph_cfg.pairwise_method == "pairs":
                        r(
                            """
                            emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                            res <- pairs(emm, adjust=p_adjust)
                            df_res <- as.data.frame(res)
                            """
                        )
                    elif ph_cfg.pairwise_method == "contrast":
                        r(
                            """
                            emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                            res <- contrast(emm, method=pairwise_contrast_method, adjust=p_adjust)
                            df_res <- as.data.frame(res)
                            """
                        )
                    else:
                        raise ValueError(
                            f"Invalid posthoc.pairwise_method={ph_cfg.pairwise_method!r}. "
                            "Use 'pairs' or 'contrast'."
                        )
            else:
                # Numeric slope tests
                spec_formula, num_var = instr
                ro.globalenv["specs"] = spec_formula
                ro.globalenv["var"] = num_var
                r(
                    """
                    emt <- suppressMessages(emtrends(final_model, specs=as.formula(specs), var=var))
                    res <- test(emt, adjust=p_adjust)
                    df_res <- as.data.frame(res)
                    """
                )

            # Keep spec variables as character columns
            vars_in_sp = extract_variables(sp)
            df_res_names = set(r("names(df_res)"))
            for v in vars_in_sp:
                if v in df_res_names:
                    r(f"df_res${v} <- as.character(df_res${v})")

            # Convert df_res to pandas.
            df_res_obj = ro.r["df_res"]
            if isinstance(df_res_obj, pd.DataFrame):
                df_res_pd = df_res_obj.copy()
            else:
                from rpy2.robjects.conversion import localconverter

                with localconverter(ro.default_converter + pandas2ri.converter):
                    df_res_pd = ro.conversion.rpy2py(df_res_obj)

            if print_info:
                print("\n" + sp)
                print(df_res_pd)

            results[sp] = df_res_pd

        if not results:
            warnings.warn("No valid specs.")

        if lrt or return_model_info:
            return LmerTestsResult(posthoc=results, lrt=lrt_df_pd, model_info=model_info)

        return results

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------
    def lmer_selection(
        self,
        full_model: str,
        numeric: Optional[Sequence[str]] = None,
        *,
        crit: Optional[str] = None,
        random_crit: Optional[str] = "BIC",
        fixed_crit: Optional[str] = "LRT",
        include: Optional[Union[str, Sequence[str]]] = None,
        print_info: bool = True,
    ) -> str:
        """Select a mixed model via backward selection using R's buildmer.

        Parameters
        ----------
        full_model:
            Full R-style model formula as a string.
        numeric:
            List of variable names that should be treated as numeric.
            The dependent variable is always treated as numeric.
        crit:
            Overall criterion for both random and fixed effects.
            If provided, `random_crit` and `fixed_crit` are ignored.
        random_crit:
            Criterion for random-effect selection.
            Common options include: ``'BIC'``, ``'LRT'``,
            ``'AIC'``, ``'AICc'``, ``'p-value'``.
            If None, random effects are not selected.
        fixed_crit:
            Criterion for fixed-effect selection.
            Common options include: ``'BIC'``, ``'LRT'``,
            ``'AIC'``, ``'AICc'``, ``'p-value'``.
            If None, fixed effects are not selected.
        include:
            One or more variable names or terms to always include
            in the selected model (e.g. intercepts, main effects).
        print_info:
            If True, print progress and results to stdout.

        Returns
        -------
        str
            The selected model formula as a string.
        """

        df_in = self._as_dataframe()
        ctx = self._get_r_context(["lme4", "buildmer"])
        pandas2ri, r, ro = ctx.pandas2ri, ctx.r, ctx.ro

        full_model = full_model.replace(" ", "")
        if "~" not in full_model:
            raise ValueError(f'Invalid model formula "{full_model}": missing "~".')

        numeric_list = self._dedupe_preserve_order(list(numeric) if numeric is not None else [])
        y = full_model.split("~", 1)[0]
        if y not in numeric_list:
            numeric_list.append(y)

        all_vars = extract_variables(full_model)
        missing = [v for v in all_vars if v not in df_in.columns]
        if missing:
            raise ValueError(f"Variables not found in DataFrame columns: {missing}")

        df = df_in.loc[:, sorted(all_vars)].copy()
        ro.globalenv["df"] = pandas2ri.py2rpy(df)

        factor_vars = sorted(set(all_vars) - set(numeric_list))
        if factor_vars:
            r("\n".join([f"df${col} <- as.factor(df${col})" for col in factor_vars]))

        if crit is not None:
            random_crit, fixed_crit = None, None

        if include is None:
            include_list: List[str] = []
        else:
            include_list = self._normalize_str_list(include)

        ro.globalenv["full_model"] = full_model
        ro.globalenv["include"] = include_list

        r(
            """
            ff <- as.formula(full_model)

            fixed_terms <- unique(c(attr(terms(nobars(ff)), "term.labels"), unlist(include)))
            fixed <- as.formula(paste("~", paste(fixed_terms, collapse = " + ")))

            random_terms <- sapply(findbars(ff), function(term) paste0("(", deparse(term), ")"))
            random_terms <- unique(c(random_terms, unlist(include)))

            if (length(random_terms) == 0) {
                random <- as.formula("~1")
            } else {
                random <- as.formula(paste("~", paste(random_terms, collapse = " + ")))
            }
            """
        )

        if random_crit is not None:
            ro.globalenv["random_crit"] = random_crit
            r(
                """
                selmod <- buildmer(
                    ff,
                    data=df,
                    buildmerControl=list(
                        direction="backward",
                        crit=random_crit,
                        include=fixed,
                        singular.ok=TRUE,
                        quiet=TRUE
                    )
                )
                ff <- formula(selmod)
                """
            )

        if fixed_crit is not None:
            ro.globalenv["fixed_crit"] = fixed_crit
            r(
                """
                selmod <- buildmer(
                    ff,
                    data=df,
                    buildmerControl=list(
                        direction="backward",
                        crit=fixed_crit,
                        include=random,
                        singular.ok=TRUE,
                        quiet=TRUE
                    )
                )
                ff <- formula(selmod)
                """
            )

        if crit is not None:
            ro.globalenv["crit"] = crit
            r(
                """
                include_formula <- if (length(include) == 0) "~1" else paste0("~", paste(include, collapse="+"))
                selmod <- buildmer(
                    ff,
                    data=df,
                    buildmerControl=list(
                        direction="backward",
                        crit=crit,
                        include=include_formula,
                        quiet=TRUE
                    )
                )
                ff <- formula(selmod)
                """
            )

        if print_info:
            print(f"Selected model: {r('ff')}")
            if include_list:
                print(f"(forced include: {', '.join(include_list)})")

        return str(r("ff")).replace("\n", "")

    # ------------------------------------------------------------------
    # Cohen's d effect sizes
    # ------------------------------------------------------------------
    def cohend(
        self,
        *,
        control_group: str = "HC",
        data_col: str = "Y",
        data_index: int = -1,
        group_col: str = "group",
        sensor_col: str = "sensor",
        min_n: int = 3,
        drop_zeros: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Compute Cohen's d for each non-control group vs control, per sensor.

        Parameters
        ----------
        control_group:
            Name of the control group level within `group_col`.
        data_col:
            Name of the data column containing numeric values or sequences.
        data_index:
            If `data_col` contains sequences (e.g., lists or arrays),
            the index within each sequence to use for effect-size computation.
            If -1, uses the entire value as-is.
        group_col:
            Name of the grouping column.
        sensor_col:
            Name of the sensor/channel column.
        min_n:
            Minimum number of samples per group required to compute Cohen's d.
            If either group has fewer than `min_n` samples for a given sensor,
            Cohen's d is set to NaN for that sensor.
        drop_zeros:
            If True, drop rows where `data_col` is zero before computing effect sizes.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dict of pandas DataFrames keyed by comparison name
            (e.g. ``"PatientvsHC"``), each containing columns:

            - `sensor`: Sensor name.
            - `d`: Cohen's d effect size.
        """

        df_in = self._as_dataframe()

        for col in (group_col, sensor_col, data_col):
            if col not in df_in.columns:
                raise ValueError(f'The column "{col}" is not in the DataFrame.')

        df = df_in[[group_col, sensor_col, data_col]].copy()

        if data_index >= 0:
            df[data_col] = df[data_col].apply(
                lambda x: x[data_index] if isinstance(x, (list, tuple, np.ndarray)) else x
            )

        if drop_zeros:
            df = df[df[data_col] != 0]

        groups = [g for g in df[group_col].dropna().unique().tolist() if g != control_group]
        results: Dict[str, pd.DataFrame] = {}

        for g in groups:
            df_pair = df[df[group_col].isin([control_group, g])]

            out_rows: List[Tuple[str, float]] = []
            for sensor, df_s in df_pair.groupby(sensor_col, sort=False):
                x = df_s.loc[df_s[group_col] == g, data_col].to_numpy(dtype=float)
                y = df_s.loc[df_s[group_col] == control_group, data_col].to_numpy(dtype=float)

                if len(x) >= min_n and len(y) >= min_n:
                    mean_x, mean_y = np.nanmean(x), np.nanmean(y)
                    std_x, std_y = np.nanstd(x, ddof=1), np.nanstd(y, ddof=1)
                    pooled = np.sqrt(
                        ((len(x) - 1) * std_x**2 + (len(y) - 1) * std_y**2) / (len(x) + len(y) - 2)
                    )
                    d = (mean_x - mean_y) / pooled if pooled != 0 else np.nan
                else:
                    d = np.nan

                out_rows.append((sensor, d))

            results[f"{g}vs{control_group}"] = pd.DataFrame(out_rows, columns=[sensor_col, "d"])

        return results

    # ------------------------------------------------------------------
    # EEG plotting using MNE
    # ------------------------------------------------------------------

    @staticmethod
    def _make_eeg_info(
        ch_names: List[str],
        montage: str,
        position_offset: Tuple[float, float],
        ch_type: str = "eeg",
    ) -> Any:
        mne = tools.dynamic_import("mne")

        if len(position_offset) != 2:
            raise ValueError("position_offset must be a tuple/list of length 2: (y_offset, z_offset).")

        info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=[ch_type] * len(ch_names))
        std_montage = mne.channels.make_standard_montage(montage)

        ch_pos = {
            ch: std_montage.get_positions()["ch_pos"][ch]
            for ch in ch_names
            if ch in std_montage.get_positions()["ch_pos"]
        }

        if not ch_pos:
            raise ValueError(f"None of the channels {ch_names} found in montage '{montage}'.")

        y_off, z_off = position_offset
        for ch in ch_pos:
            ch_pos[ch] = ch_pos[ch].copy()
            ch_pos[ch][1] += y_off
            ch_pos[ch][2] += z_off

        new_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info.set_montage(new_montage)
        return info
    
    def eeg_topomap(
        self,
        values=None,
        ch_names=None,
        *,
        position_offset,
        sensor_col="sensor",
        data_col="data",
        data_index=-1,
        montage="standard_1005",
        ch_type="eeg",
        sphere=0.28,
        cmap="jet",
        vmin=None,
        vmax=None,
        colorbar=True,
        colorbar_fmt="%0.2f",
        colorbar_label=None,
        colorbar_label_fontsize=None,
        colorbar_tick_fontsize=8,
        sensors=None,
        outlines=None,
        contours=None,
        res=300,
        extrapolate=None,
        image_interp=None,
        axes=None,
        show=True,
        title=None,
        names=None,
    ):
        if not tools.ensure_module("mne"):
            raise ImportError("mne is required for eeg_topomap but is not installed.")

        import matplotlib.pyplot as plt
        mne_viz = tools.dynamic_import("mne", "viz")

        # Paso A — Resolver datos y nombres de canales
        if values is None:
            df = self._as_dataframe()
            if sensor_col not in df.columns:
                raise ValueError(f"Column '{sensor_col}' not found in data.")
            if data_col not in df.columns:
                raise ValueError(f"Column '{data_col}' not found in data.")
            if df.empty:
                raise ValueError("DataFrame is empty.")
            ch_names_in = []
            data_vals = []
            for sensor, group in df.groupby(sensor_col, sort=False):
                raw = group[data_col].values
                if data_index == -1:
                    if len(raw) != 1:
                        raise ValueError(
                            f"data_index=-1 expects a scalar per sensor, but sensor '{sensor}' has {len(raw)} rows."
                        )
                    val = float(raw[0])
                elif data_index is None:
                    val = float(np.mean(raw))
                else:
                    if data_index >= len(raw):
                        raise IndexError(
                            f"data_index={data_index} out of range for sensor '{sensor}' with {len(raw)} rows."
                        )
                    val = float(raw[data_index])
                ch_names_in.append(str(sensor))
                data_vals.append(val)
            data_arr = np.asarray(data_vals, dtype=float)
        else:
            if isinstance(values, pd.Series):
                ch_names_in = list(values.index.astype(str))
                data_arr = values.to_numpy(dtype=float)
            elif isinstance(values, Mapping):
                ch_names_in = [str(k) for k in values.keys()]
                data_arr = np.asarray(list(values.values()), dtype=float)
            else:
                data_arr = np.asarray(values, dtype=float)
                if ch_names is None:
                    raise ValueError("When values is array-like, ch_names is required.")
                ch_names_in = [str(c) for c in ch_names]

        if data_arr.ndim != 1:
            raise ValueError(f"values must be 1D, got shape {data_arr.shape}.")
        if len(ch_names_in) != data_arr.shape[0]:
            raise ValueError(f"Length mismatch: {len(ch_names_in)} channel names vs {data_arr.shape[0]} values.")

        # Paso B — Validar position_offset
        if len(position_offset) != 2 or not all(isinstance(v, (int, float)) for v in position_offset):
            raise ValueError("position_offset must be a tuple/list of two numbers: (y_offset, z_offset).")

        # Paso C — Crear Info
        info_use = self._make_eeg_info(ch_names_in, montage, position_offset, ch_type)

        # Paso D — All-zeros
        if np.all(data_arr == 0):
            cmap = "Greys"
            colorbar = False
            vmin = vmax = 0.0

        # Paso E — vmin/vmax
        if vmin is None:
            vmin = float(np.nanmin(data_arr))
        if vmax is None:
            vmax = float(np.nanmax(data_arr))
        if vmin > vmax:
            raise ValueError(f"vmin ({vmin}) must be <= vmax ({vmax}).")

        # Paso F — Plot
        # Match the legacy plots.py call by default. Optional MNE kwargs are
        # forwarded only when explicitly provided, because forcing them can
        # change interpolation/contours across MNE versions.
        topomap_kwargs = {"sphere": sphere, "res": res}
        if sensors is not None:
            topomap_kwargs["sensors"] = sensors
        if outlines is not None:
            topomap_kwargs["outlines"] = outlines
        if contours is not None:
            topomap_kwargs["contours"] = contours
        if extrapolate is not None:
            topomap_kwargs["extrapolate"] = extrapolate
        if image_interp is not None:
            topomap_kwargs["image_interp"] = image_interp

        im, _ = mne_viz.plot_topomap(
            data_arr, info_use, axes=axes, cmap=cmap, show=False,
            **topomap_kwargs,
        )
        im.set_clim(vmin, vmax)

        # Paso G — Colorbar manual
        if colorbar:
            mid = vmin + (vmax - vmin) / 2
            cbar = plt.colorbar(
                im, ax=axes if axes is not None else im.axes,
                fraction=0.046, pad=0.0, format=colorbar_fmt, ticks=[vmin, mid, vmax],
            )
            cbar.ax.tick_params(labelsize=colorbar_tick_fontsize)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=colorbar_label_fontsize)

        # Paso H — Título y show
        if title:
            (axes if axes is not None else im.axes).set_title(title)
        if show:
            plt.show()

        return im

    # --------------
    # MEG plotting
    # --------------
    def meg_surface(
        self,
        values: Union[Sequence[float], np.ndarray, pd.Series],
        *,
        atlas: str = "dk",
        subject: str = "fsaverage",
        subjects_dir: Optional[str] = None,
        surface: str = "inflated",
        hemisphere: str = "both",
        views: Union[str, Sequence[str]] = "lat",
        cmap: str = "coolwarm",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        center: Optional[float] = None,
        alpha: float = 1.0,
        smoothing_steps: Optional[int] = 5,
        colorbar: bool = True,
        colorbar_kwargs: Optional[Mapping[str, Any]] = None,
        cortex: Union[str, Tuple[Any, ...]] = "low_contrast",
        background: str = "white",
        size: Union[int, Tuple[int, int]] = (1000, 600),
        show: bool = True,
        axes: Optional[Any] = None,
        auto_fetch: bool = True,
        hcp_accept: bool = False,
        verbose: Optional[Union[bool, str, int]] = None,
        **brain_kwargs: Any,
    ) -> Tuple[Any, Any, List[str]]:
        """Plot parcel-wise MEG values on a cortical surface using MNE.

        This function is atlas-only by design. It maps one value per cortical
        parcel to the corresponding atlas labels, expands to vertex-wise data,
        and renders a brain surface plot.

        The `atlas` argument accepts any parcellation name readable by
        ``mne.read_labels_from_annot(subject=..., parc=atlas, ...)``. A few common
        aliases are normalized (e.g., ``"dk"`` -> ``"aparc"``).

        Parameters
        ----------
        values:
            1D array-like with one value per atlas parcel. Order is:
            sorted left-hemisphere labels followed by sorted right-hemisphere labels
            (after removing ``unknown``, ``corpuscallosum``, and ``medialwall`` labels).
        atlas:
            Atlas/parcellation name (MNE/FreeSurfer annotation name).
        subject:
            FreeSurfer subject name for which the atlas is defined.
        subjects_dir:
            Directory where subject surfaces and atlas files are read/downloaded.
            If None, MNE default location is used.
        surface:
            Surface name (e.g., ``"inflated"``, ``"pial"``, ``"white"``).
        hemisphere:
            Hemisphere display mode: ``"lh"``, ``"rh"``, ``"both"``, or ``"split"``.
        views:
            Brain view(s), passed to MNE Brain.
        cmap:
            Colormap name.
        vmin, vmax:
            Color limits. If None, inferred from `values`.
        center:
            Optional center for diverging colormaps.
        alpha:
            Overlay opacity.
        smoothing_steps:
            Smoothing steps passed to ``Brain.add_data`` before screenshot rendering.
        colorbar:
            Whether to show colorbar.
        colorbar_kwargs:
            Optional keyword arguments forwarded to Matplotlib ``fig.colorbar``
            (for example ``{"ticks": [-1, 0, 1]}``).
        cortex:
            Cortex style passed to MNE Brain.
        background:
            Background color.
        size:
            Figure size passed to MNE Brain.
        show:
            Whether to show the Matplotlib figure immediately.
        axes:
            Optional Matplotlib axes where the rendered brain image will be drawn.
            If None, a new figure and axes are created.
        auto_fetch:
            If True, auto-fetches `fsaverage` when needed. For non-fsaverage subjects,
            this function expects the subject to already exist in `subjects_dir`.
        hcp_accept:
            Required only when downloading HCP-MMP files and they are not already present.
            Set True to accept HCP-MMP license terms.
        verbose:
            MNE verbosity setting.
        **brain_kwargs:
            Extra keyword arguments forwarded to MNE Brain constructor.

        Returns
        -------
        fig, ax, label_order:
            Matplotlib figure/axes and the label-name order used to map input values.
        """
        if not tools.ensure_module("mne"):
            raise ImportError("mne is required for atlas-based MEG surface plotting.")
        if not tools.ensure_module("nibabel"):
            raise ImportError(
                "nibabel is required for meg_surface (MNE needs it to read cortical surfaces). "
                "Install it with: pip install nibabel"
            )
        mne = tools.dynamic_import("mne")
        from pathlib import Path

        if hemisphere not in {"lh", "rh", "both", "split"}:
            raise ValueError("hemisphere must be one of {'lh', 'rh', 'both', 'split'}.")

        atlas_norm = atlas.strip().lower()
        alias_to_parc: Dict[str, str] = {
            "dk": "aparc",
            "desikan": "aparc",
            "desikan-killiany": "aparc",
            "aparc": "aparc",
            "destrieux": "aparc.a2009s",
            "aparc.a2009s": "aparc.a2009s",
            "hcp-mmp": "HCPMMP1",
            "hcp": "HCPMMP1",
            "glasser": "HCPMMP1",
            "hcpmmp1": "HCPMMP1",
            "hcp-mmp-combined": "HCPMMP1_combined",
            "glasser-combined": "HCPMMP1_combined",
            "hcpmmp1_combined": "HCPMMP1_combined",
            "aparc-sub": "aparc_sub",
            "khan": "aparc_sub",
            "aparc_sub": "aparc_sub",
        }
        parc = alias_to_parc.get(atlas_norm, atlas.strip())

        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("values must contain at least one element.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("values must contain only finite numeric values.")

        subjects_dir_path = Path(subjects_dir).expanduser() if subjects_dir is not None else None
        if subjects_dir_path is not None:
            subjects_dir_path.mkdir(parents=True, exist_ok=True)

        if auto_fetch and subject == "fsaverage":
            subjects_dir_resolved = Path(
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir_path, verbose=verbose)
            ).parent
        else:
            if subjects_dir_path is None:
                cfg = mne.get_config("SUBJECTS_DIR", default=None)
                subjects_dir_resolved = Path(cfg).expanduser() if cfg else None
            else:
                subjects_dir_resolved = subjects_dir_path
            if subjects_dir_resolved is None:
                raise ValueError(
                    "subjects_dir is required when auto_fetch=False and SUBJECTS_DIR is not configured."
                )
        subject_dir = subjects_dir_resolved / subject
        if not subject_dir.exists():
            raise ValueError(
                f"Subject directory not found: {subject_dir}. "
                "Provide a valid subject or enable auto_fetch with subject='fsaverage'."
            )

        label_dir = subject_dir / "label"
        if parc in {"HCPMMP1", "HCPMMP1_combined"} and subject == "fsaverage":
            lh_annot = label_dir / f"lh.{parc}.annot"
            rh_annot = label_dir / f"rh.{parc}.annot"
            if not (lh_annot.exists() and rh_annot.exists()):
                if not hcp_accept:
                    raise ValueError(
                        f"Parcellation '{parc}' is not available locally for fsaverage. "
                        "Set hcp_accept=True to accept the license and download it."
                    )
                mne.datasets.fetch_hcp_mmp_parcellation(
                    subjects_dir=subjects_dir_resolved,
                    combine=True,
                    accept=True,
                    verbose=verbose,
                )
        elif parc == "aparc_sub" and subject == "fsaverage":
            mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir_resolved, verbose=verbose)

        try:
            labels = mne.read_labels_from_annot(
                subject=subject,
                parc=parc,
                hemi="both",
                subjects_dir=subjects_dir_resolved,
                sort=True,
                verbose=verbose,
            )
        except Exception as exc:
            raise ValueError(
                f"Could not read atlas/parcellation '{parc}' for subject '{subject}'. "
                f"Ensure lh.{parc}.annot and rh.{parc}.annot exist in "
                f"{subject_dir / 'label'}."
            ) from exc

        def _is_plot_label(name: str) -> bool:
            n = name.lower()
            excluded = ("unknown", "corpuscallosum", "medialwall")
            return not any(e in n for e in excluded)

        labels = [lab for lab in labels if _is_plot_label(lab.name)]
        labels_lh = [lab for lab in labels if lab.hemi == "lh"]
        labels_rh = [lab for lab in labels if lab.hemi == "rh"]
        label_order = [lab.name for lab in labels_lh + labels_rh]

        if arr.shape[0] != len(label_order):
            raise ValueError(
                f"Atlas '{parc}' expects {len(label_order)} values "
                f"(LH {len(labels_lh)} + RH {len(labels_rh)}), got {arr.shape[0]}."
            )

        if vmin is None:
            vmin = float(np.nanmin(arr))
        if vmax is None:
            vmax = float(np.nanmax(arr))
        if vmin > vmax:
            raise ValueError(f"vmin ({vmin}) must be <= vmax ({vmax}).")

        lh_surface_path = subject_dir / "surf" / f"lh.{surface}"
        rh_surface_path = subject_dir / "surf" / f"rh.{surface}"
        if not lh_surface_path.exists() or not rh_surface_path.exists():
            raise ValueError(
                f"Surface '{surface}' not found in subject surf directory "
                f"({subject_dir / 'surf'})."
            )
        n_lh_vertices = mne.read_surface(str(lh_surface_path), verbose=verbose)[0].shape[0]
        n_rh_vertices = mne.read_surface(str(rh_surface_path), verbose=verbose)[0].shape[0]

        data_lh = np.full(n_lh_vertices, np.nan, dtype=float)
        data_rh = np.full(n_rh_vertices, np.nan, dtype=float)
        lh_vals = arr[: len(labels_lh)]
        rh_vals = arr[len(labels_lh):]

        for label, val in zip(labels_lh, lh_vals):
            data_lh[label.vertices] = float(val)
        for label, val in zip(labels_rh, rh_vals):
            data_rh[label.vertices] = float(val)

        if np.isnan(data_lh).any():
            data_lh = np.nan_to_num(data_lh, nan=vmin)
        if np.isnan(data_rh).any():
            data_rh = np.nan_to_num(data_rh, nan=vmin)

        import inspect
        import matplotlib.pyplot as plt
        from matplotlib import cm as mpl_cm
        from matplotlib.colors import Normalize

        fmid = center if center is not None else (vmin + vmax) / 2.0

        Brain = mne.viz.get_brain_class()
        brain_sig = inspect.signature(Brain)
        accepted_brain_kwargs = set(brain_sig.parameters)
        brain_args_common: Dict[str, Any] = dict(
            cortex=cortex,
            size=size,
            background=background,
            subjects_dir=str(subjects_dir_resolved),
            views=views,
            show=False,
        )
        brain_args_common.update(brain_kwargs)
        brain_args_common = {k: v for k, v in brain_args_common.items() if k in accepted_brain_kwargs}

        def _render_hemi(hemi_render: str, data_hemi: np.ndarray) -> np.ndarray:
            brain = Brain(subject, hemi_render, surface, **brain_args_common)
            add_sig = inspect.signature(brain.add_data)
            accepted_add_kwargs = set(add_sig.parameters)
            add_kwargs: Dict[str, Any] = dict(
                fmin=vmin,
                fmid=fmid,
                fmax=vmax,
                center=center,
                colormap=cmap,
                alpha=alpha,
                smoothing_steps=smoothing_steps,
                colorbar=False,
                verbose=verbose,
            )
            add_kwargs = {k: v for k, v in add_kwargs.items() if k in accepted_add_kwargs}
            brain.add_data(data_hemi, hemi=hemi_render, **add_kwargs)
            # Some MNE/pyvista versions need an explicit render before screenshot.
            if hasattr(brain, "_renderer") and hasattr(brain._renderer, "plotter"):
                try:
                    brain._renderer.plotter.render()
                except Exception:
                    pass
            try:
                img = brain.screenshot(mode="rgb", time_viewer=False)
            except TypeError:
                img = brain.screenshot()
            finally:
                if hasattr(brain, "close"):
                    brain.close()
            return np.asarray(img)

        if hemisphere == "lh":
            img = _render_hemi("lh", data_lh)
        elif hemisphere == "rh":
            img = _render_hemi("rh", data_rh)
        else:
            img_lh = _render_hemi("lh", data_lh)
            img_rh = _render_hemi("rh", data_rh)
            h = min(img_lh.shape[0], img_rh.shape[0])
            img_lh = img_lh[:h, ...]
            img_rh = img_rh[:h, ...]
            gap = np.full((h, 16, img_lh.shape[2]), 255, dtype=img_lh.dtype)
            img = np.concatenate([img_lh, gap, img_rh], axis=1)

        if axes is None:
            if isinstance(size, tuple):
                figsize = (max(6.0, size[0] / 150.0), max(4.0, size[1] / 150.0))
            else:
                figsize = (8.0, 5.0)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            ax = axes
            fig = ax.figure
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        ax.imshow(img)
        ax.set_axis_off()

        if colorbar:
            norm = Normalize(vmin=vmin, vmax=vmax)
            mappable = mpl_cm.ScalarMappable(norm=norm, cmap=mpl_cm.get_cmap(cmap))
            mappable.set_array(arr)
            cb_kwargs = dict(colorbar_kwargs) if colorbar_kwargs is not None else {}
            if "ticks" not in cb_kwargs:
                cb_kwargs["ticks"] = np.linspace(vmin, vmax, 3)
            fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03, **cb_kwargs)

        if show:
            plt.show()

        return fig, ax, label_order
