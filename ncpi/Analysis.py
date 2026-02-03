"""
Notes
-----
- Mixed model functionality depends on **rpy2** and the relevant **R packages**.
  `lmer_tests` requires R packages: lme4, emmeans
  `lmer_selection` requires R packages: lme4, buildmer

"""

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
    """Statistical analysis and EEG visualization helper.

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
        """Import rpy2 and ensure required R packages are available.

        Parameters
        ----------
        packages:
            R package names to load with `require()`.

        Returns
        -------
        _RContext
            Container with rpy2 objects: `pandas2ri`, `r`, and `robjects`.

        Raises
        ------
        ImportError
            If rpy2 is not installed.
        RuntimeError
            If an R package is missing (raised by R and surfaced through rpy2).
        """
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

        Workflow
        --------
        1) Fit one or more candidate models using R (`lm` or `lmer`)
        2) If multiple candidates are given, select the one with the lowest BIC
        3) Interpret `specs`:
           - factor-only specs -> `emmeans()` + contrasts/pairs
           - specs with exactly one numeric variable -> `emtrends()` + `test()`
        4) Convert results back to pandas DataFrames.

        Parameters
        ----------
        models:
            One or more R-style model formulas.

            If multiple are provided, all are fitted and the one with smallest BIC is used.
            All formulas must share the same dependent variable (left side of `~`).

        group_col:
            Name of the primary grouping column. Required when `specs` is None.

        control_group:
            If provided, and a spec corresponds to the primary group term, comparisons are
            computed using a treatment-vs-control contrast (see `posthoc.treatment_method`).
            Otherwise, all-pairs comparisons are computed.

        numeric:
            List of variables to treat as numeric. All other variables are converted to factors
            in R. The dependent variable is always treated as numeric.

        specs:
            Post-hoc specifications. Each spec is a compact string that may include:
            - `:` for interactions
            - `|` for conditioning, e.g. `sensor|group` means compare sensors within each group

            If the spec contains no numeric variables -> `emmeans` comparisons.
            If it contains exactly one numeric variable -> `emtrends` slope tests.

            Specs that refer to variables not present in the selected model's fixed effects are
            skipped (with a message if `print_info=True`).

        posthoc:
            Optional :class:`PosthocConfig` or mapping configuring:
            - p-value adjustment (`adjust`)
            - all-pairs method (`pairwise_method`, `pairwise_contrast_method`)
            - treatment-vs-control method and reference (`treatment_method`, `treatment_ref`)

        print_info:
            Print model selection and per-spec results.

        lrt:
            If True, perform a likelihood-ratio test (LRT) between the selected model
            and a reduced model with terms in `lrt_drop` removed.

        lrt_drop:
            List of terms to drop from the selected model when constructing the reduced model
            for LRT. Required if `lrt=True`.

        return_model_info:
            If True, return extra model-selection info such as BICs and the selected formula.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Mapping from spec string to its results DataFrame.

        Warnings
        --------
        Warns if no valid specs remain after filtering.

        Raises
        ------
        TypeError
            If `self.data` is not a DataFrame.
        ValueError
            For invalid formulas, missing columns, or inconsistent grouping config.
        ImportError
            If rpy2 is missing.
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

        # Fit candidate models
        for i, formula in enumerate(models_list):
            fit_fun = "lmer" if "|" in formula else "lm"
            r(f"m{i} <- {fit_fun}({formula}, data=df)")

        # BIC-based selection if more than one candidate
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

        # Fixed effects present in selected model
        selmod_fixed = set(ro.r('attr(terms(nobars(final_model)), "term.labels")'))

        # Normalize specs (default to group_col)
        if specs is None:
            specs_list = [group_col]  # type: ignore[list-item]
        else:
            specs_list = self._normalize_str_list(specs)

        # Parse specs into aligned instructions
        SpecInstruction = Union[str, Tuple[str, str]]  # "~spec" OR ("~spec_without_numeric", "numeric_var")
        spec_instructions: List[Tuple[str, SpecInstruction]] = []

        numeric_set = set(numeric_list)
        for sp in specs_list:
            vars_in_sp = extract_variables(sp)

            if any(v not in selmod_fixed for v in vars_in_sp):
                if print_info:
                    print(f'(!) Specs "{sp}" skipped: variable not present as fixed effect.')
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
                        df_res$contrast <- as.character(res@grid$contrast)
                        """
                    )
                else:
                    if ph_cfg.pairwise_method == "pairs":
                        r(
                            """
                            emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                            res <- pairs(emm, adjust=p_adjust)
                            df_res <- as.data.frame(res)
                            df_res$contrast <- as.character(res@grid$contrast)
                            """
                        )
                    elif ph_cfg.pairwise_method == "contrast":
                        r(
                            """
                            emm <- suppressMessages(emmeans(final_model, specs=as.formula(specs)))
                            res <- contrast(emm, method=pairwise_contrast_method, adjust=p_adjust)
                            df_res <- as.data.frame(res)
                            df_res$contrast <- as.character(res@grid$contrast)
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
            # Depending on rpy2 conversion settings, ro.r['df_res'] may already be a pandas.DataFrame.
            df_res_obj = ro.r['df_res']
            if isinstance(df_res_obj, pd.DataFrame):
                df_res_pd = df_res_obj.copy()
            else:
                # Use a local converter to avoid relying on deprecated global conversion.
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
            Starting model formula, e.g. ``'Y ~ group*epoch + (1|id)'``.
        numeric:
            Variables to treat as numeric (others become factors).
        crit:
            If set, a single criterion used for the entire selection procedure.
            When provided, `random_crit` and `fixed_crit` are ignored.
        random_crit:
            Criterion for random-effect structure selection (default 'BIC').
        fixed_crit:
            Criterion for fixed-effect selection after random selection (default 'LRT').
            Set to None to skip fixed-effect selection.
        include:
            One or more terms to force-include (never dropped).
        print_info:
            Print selection summary.

        Returns
        -------
        str
            The selected model formula (single-line string).
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

        Notes
        -----
        - d is computed using pooled standard deviation (unbiased sample SD, ddof=1).
        - If `pooled` is 0 or sample sizes are too small, d is returned as NaN.

        Returns
        -------
        dict[str, pandas.DataFrame]
            Mapping ``"{group}vs{control_group}"`` -> DataFrame with columns [sensor_col, "d"].
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
            for sensor, df_s in df_pair.groupby(sensor_col):
                x = df_s.loc[df_s[group_col] == g, data_col].to_numpy(dtype=float)
                y = df_s.loc[df_s[group_col] == control_group, data_col].to_numpy(dtype=float)

                if len(x) >= min_n and len(y) >= min_n:
                    mean_x, mean_y = np.nanmean(x), np.nanmean(y)
                    std_x, std_y = np.nanstd(x, ddof=1), np.nanstd(y, ddof=1)
                    pooled = np.sqrt(((len(x) - 1) * std_x**2 + (len(y) - 1) * std_y**2) / (len(x) + len(y) - 2))
                    d = (mean_x - mean_y) / pooled if pooled != 0 else np.nan
                else:
                    d = np.nan

                out_rows.append((sensor, d))

            results[f"{g}vs{control_group}"] = pd.DataFrame(out_rows, columns=[sensor_col, "d"])

        return results

    # ------------------------------------------------------------------
    # EEG plotting using MNE
    # ------------------------------------------------------------------
    def eeg_topomap(
        self,
        values: Union[Sequence[float], np.ndarray, Mapping[str, float], pd.Series],
        *,
        ch_names: Optional[Sequence[str]] = None,
        montage: Union[str, Any] = "standard_1020",
        info: Optional[Any] = None,
        ch_type: str = "eeg",
        units: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: Optional[str] = None,
        sensors: bool = True,
        outlines: Union[str, dict] = "head",
        sphere: Union[str, Tuple[float, float, float, float]] = "auto",
        axes: Optional[Any] = None,
        show: bool = True,
        colorbar: bool = True,
        res: int = 64,
        extrapolate: str = "auto",
        image_interp: str = "cubic",
    ):
        """Plot an EEG topomap using MNE, adapting to any montage / channel configuration.

        Parameters
        ----------
        values:
            Data to plot. Accepted forms:

            1) 1D array-like of shape (n_channels,)
               - must provide `ch_names` with the same length

            2) Mapping ``{channel_name: value}``
               - `ch_names` is inferred from the mapping keys

            3) pandas Series
               - channel names come from the Series index

        ch_names:
            Channel names corresponding to `values` when `values` is array-like.

        montage:
            Montage definition:
            - string name (e.g., "standard_1020", "standard_1005", ...)
            - an MNE DigMontage instance

        info:
            Optional MNE Info. If provided, it is used as the basis.
            Otherwise, an Info is created with `sfreq=1.0`.

        ch_type:
            Channel type when creating Info (default "eeg").

        units:
            Optional short label. If `axes` is provided, it's placed as the axes title.

        vmin, vmax, cmap, sensors, outlines, sphere, axes, show, colorbar, res, extrapolate, image_interp:
            Forwarded to `mne.viz.plot_topomap`.

        Returns
        -------
        (im, cn)
            Image and contour objects returned by MNE.

        Raises
        ------
        ImportError
            If `mne` is not installed.
        ValueError
            If channel names and values mismatch or channels are missing from the montage.

        Tips
        ----
        - For non-standard EEG systems, create an MNE DigMontage with your exact channel positions
          and pass it via `montage=...`.
        """
        if not tools.ensure_module("mne"):
            raise ImportError("mne is required for eeg_topomap but is not installed.")

        mne = tools.dynamic_import("mne")
        mne_viz = tools.dynamic_import("mne", "viz")

        # Normalize values + channel names
        if isinstance(values, pd.Series):
            ch_names_in = list(values.index.astype(str))
            data = values.to_numpy(dtype=float)
        elif isinstance(values, Mapping):
            ch_names_in = [str(k) for k in values.keys()]
            data = np.asarray(list(values.values()), dtype=float)
        else:
            data = np.asarray(values, dtype=float)
            if ch_names is None:
                raise ValueError("When `values` is array-like, you must provide `ch_names`.")
            ch_names_in = [str(c) for c in ch_names]

        if data.ndim != 1:
            raise ValueError(f"values must be 1D (n_channels,), got shape {data.shape}.")
        if len(ch_names_in) != data.shape[0]:
            raise ValueError(f"Length mismatch: {len(ch_names_in)} channel names vs {data.shape[0]} values.")

        # Build or validate Info
        if info is None:
            info_use = mne.create_info(ch_names=ch_names_in, sfreq=1.0, ch_types=[ch_type] * len(ch_names_in))
        else:
            missing = [c for c in ch_names_in if c not in info["ch_names"]]
            if missing:
                raise ValueError(f"Provided info is missing channels: {missing}")
            info_use = info.copy()

        # Apply montage
        if montage is not None:
            try:
                if isinstance(montage, str):
                    mont = mne.channels.make_standard_montage(montage)
                else:
                    mont = montage
                info_use.set_montage(mont, match_case=False, on_missing="ignore")
            except Exception as e:
                raise ValueError(f"Failed to set montage {montage!r}: {e}") from e

        # Ensure channel order matches data
        if info_use["ch_names"] != ch_names_in:
            info_tmp = mne.create_info(
                ch_names=ch_names_in, sfreq=info_use["sfreq"], ch_types=[ch_type] * len(ch_names_in)
            )
            try:
                info_tmp.set_montage(info_use.get_montage(), match_case=False, on_missing="ignore")
            except Exception:
                pass
            info_use = info_tmp

                # MNE has changed the topomap API across versions.
        # We therefore build keyword arguments dynamically based on the installed MNE.
        import inspect

        sig = inspect.signature(mne_viz.plot_topomap)
        supported = set(sig.parameters.keys())

        kwargs = dict(
            axes=axes,
            show=show,
            cmap=cmap,
            sensors=sensors,
            outlines=outlines,
            sphere=sphere,
            res=res,
            extrapolate=extrapolate,
            image_interp=image_interp,
        )

        # Optional colorbar support (older MNE may not accept it)
        if "colorbar" in supported:
            kwargs["colorbar"] = colorbar

        # Value range support differs across versions: either (vmin, vmax) or vlim
        if "vmin" in supported or "vmax" in supported:
            if "vmin" in supported:
                kwargs["vmin"] = vmin
            if "vmax" in supported:
                kwargs["vmax"] = vmax
        elif "vlim" in supported:
            kwargs["vlim"] = (vmin, vmax)

        im, cn = mne_viz.plot_topomap(data, info_use, **kwargs)

        if units is not None and axes is not None:
            axes.set_title(units)

        return im, cn