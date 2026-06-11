"""Generate the WebUI information reference from the tooltip source text."""

from __future__ import annotations

import ast
import html
import json
import re
from collections import OrderedDict
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOLTIP_JS = ROOT / "webui/static/js/fieldHelpTooltips.js"
TEMPLATES = ROOT / "webui/templates"
APP_PY = ROOT / "webui/app.py"
OUTPUT = ROOT / "docs/tutorials/webui/webui-information-reference.html"

MODULES = OrderedDict(
    (
        ("simulation", "Simulation"),
        ("field-potential", "Field Potential"),
        ("features", "Features"),
        ("inference", "Inference"),
        ("analysis", "Analysis"),
    )
)

MODULE_DESCRIPTIONS = {
    "simulation": "Simulation setup, neural network models, parameter sweeps, and model-specific controls.",
    "field-potential": "Proxy signals, biophysical kernels, CDM/LFP computation, and M/EEG forward models.",
    "features": "Feature methods, computation settings, data loading, and electrophysiology parsing.",
    "inference": "Inverse-model training, model assets, prediction settings, and SBI controls.",
    "analysis": "Boxplots, topographic maps, and simulation-result visualization controls.",
}

MODULE_BY_TEMPLATE_PREFIX = {
    "0.": "overview",
    "1.": "simulation",
    "2.": "field-potential",
    "3.": "features",
    "4.": "inference",
    "5.": "analysis",
}

MODULE_BY_ROUTE_PREFIX = {
    "/simulation": "simulation",
    "/field_potential": "field-potential",
    "/features": "features",
    "/inference": "inference",
    "/analysis": "analysis",
}


def clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def humanize_key(key: str) -> str:
    replacements = {
        "cdm": "CDM",
        "dfa": "DFA",
        "eeg": "EEG",
        "fei": "fEI",
        "fft": "FFT",
        "gaba": "GABA",
        "hctsa": "hctsa",
        "lfp": "LFP",
        "meg": "MEG",
        "meeg": "M/EEG",
        "mse": "MSE",
        "nfft": "NFFT",
        "noverlap": "N overlap",
        "nperseg": "N per segment",
        "ou": "OU",
        "sbi": "SBI",
    }
    words = []
    for word in key.split("_"):
        words.append(replacements.get(word, word.upper() if len(word) == 1 else word.capitalize()))
    return " ".join(words)


def parse_js_object(source: str, name: str) -> list[tuple[str, str]]:
    match = re.search(rf"const {re.escape(name)} = \{{(.*?)\n    \}};", source, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find JavaScript object {name}")
    entries = []
    for key, value in re.findall(r'^\s*([A-Za-z0-9_]+): ("(?:\\.|[^"\\])*")', match.group(1), re.MULTILINE):
        entries.append((key, json.loads(value)))
    return entries


def parse_main_field_help(source: str) -> dict[str, list[tuple[str, str, str]]]:
    grouped = {module: [] for module in MODULES}
    current = "simulation"
    transitions = {
        "proxy_method": "field-potential",
        "features_n_jobs": "features",
        "training_model_name": "inference",
        "boxplot_group_by": "analysis",
    }
    for key, text in parse_js_object(source, "helpByKey"):
        current = transitions.get(key, current)
        grouped[current].append((key, humanize_key(key), text))
    return grouped


def field_group(module: str, key: str) -> str:
    if module == "simulation":
        if key in {
            "tstop", "dt", "local_num_threads", "sim_run_mode", "sim_numpy_seed",
            "grid_start", "grid_step", "grid_end", "sim_repetitions",
        }:
            return "Simulation and parameter-sweep controls"
        if key in {"areas", "x", "n_x", "c_m_x", "tau_m_x", "e_l_x", "model"}:
            return "Shared model structure and population parameters"
        if key in {"c_yx", "j_yx", "delay_yx", "tau_syn_yx", "n_ext", "nu_ext", "j_ext"}:
            return "Shared connectivity and external-input parameters"
        if key == "four_area_local_editor":
            return "Four-area model parameters"
        return "Shared spatial and neuron-dynamics parameters"
    if module == "field-potential":
        if key in {"proxy_method", "sim_step", "proxy_decimation_factor", "bin_size", "excitatory_only"}:
            return "Proxy-signal configuration"
        if key.startswith("cdm_"):
            return "CDM and LFP convolution"
        if key.startswith("meeg_"):
            return "M/EEG forward modelling"
        return "Biophysical kernel construction"
    if module == "features":
        if key.startswith("features_"):
            return "Feature-computation settings"
        if key.startswith("catch22_"):
            return "catch22 parameters"
        if key.startswith("specparam_"):
            return "Spectral parameterization"
        if key.startswith("dfa_"):
            return "Detrended fluctuation analysis (DFA)"
        if key.startswith("fei_"):
            return "Functional excitation-inhibition ratio (fEI)"
        if key.startswith("custom_"):
            return "Custom feature functions"
        return "Electrophysiology dataset parser"
    if module == "inference":
        if key.startswith("training_") or key == "sbi_prior":
            return "Inverse-model training"
        if key in {"features_source_mode", "inference_model_assets_source", "inference_scaler"}:
            return "Prediction data and model assets"
        return "Prediction and posterior-sampling controls"
    if module == "analysis":
        if key.startswith("boxplot_"):
            return "Boxplot configuration"
        if key.startswith("topomap_"):
            return "Topographic-map configuration"
        return "Simulation-result plots"
    raise KeyError(module)


def parse_page_help(source: str) -> dict[str, list[tuple[str, str]]]:
    grouped = {module: [] for module in MODULES}
    match = re.search(r"const pageHelp = \[(.*?)\n    \];", source, re.DOTALL)
    if not match:
        raise RuntimeError("Could not find pageHelp")
    for path, text in re.findall(
        r'\[("(?:\\.|[^"\\])*"),\s*("(?:\\.|[^"\\])*")\]',
        match.group(1),
    ):
        path, text = json.loads(path), json.loads(text)
        module = next((value for prefix, value in MODULE_BY_ROUTE_PREFIX.items() if path.startswith(prefix)), None)
        if module:
            grouped[module].append((path, text))
    return grouped


def parse_dashboard_help() -> dict[str, list[tuple[str, str]]]:
    source = (TEMPLATES / "0.dashboard.html").read_text(encoding="utf-8")
    grouped = {module: [] for module in MODULES}
    for module, title in MODULES.items():
        tooltip_id = f"{module}-module-tooltip"
        match = re.search(rf'id="{re.escape(tooltip_id)}"[^>]*>(.*?)</span>', source, re.DOTALL)
        if match:
            grouped[module].append((f"{title} module", clean_text(match.group(1))))
    return grouped


def parse_menu_choices() -> dict[str, list[tuple[str, str]]]:
    grouped = {module: [] for module in MODULES}
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    function_modules = {
        "simulation": "simulation",
        "field_potential": "field-potential",
        "features": "features",
        "inference": "inference",
    }
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in function_modules:
            continue
        module = function_modules[node.name]
        assignment = next(
            (
                statement
                for statement in node.body
                if isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "options" for target in statement.targets)
                and isinstance(statement.value, ast.List)
            ),
            None,
        )
        if not assignment:
            continue
        for item in assignment.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            values = {
                key.value: value.value
                for key, value in zip(item.keys, item.values)
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            }
            if values.get("title") and values.get("description"):
                grouped[module].append((values["title"], values["description"]))
    return grouped


class StaticHelpParser(HTMLParser):
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.captures: list[dict[str, object]] = []
        self.results: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.VOID_TAGS:
            self.depth += 1
        attributes = dict(attrs)
        help_text = attributes.get("data-help")
        if help_text and help_text != "method.help":
            self.captures.append(
                {
                    "depth": self.depth,
                    "help": help_text,
                    "label": attributes.get("aria-label") or attributes.get("title") or "",
                    "text": [],
                }
            )

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        for capture in self.captures:
            capture["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        closing = [capture for capture in self.captures if capture["depth"] == self.depth]
        for capture in closing:
            label = clean_text(str(capture["label"]) or " ".join(capture["text"]))
            self.results.append((label or "Additional information", str(capture["help"])))
            self.captures.remove(capture)
        self.depth -= 1


def template_group(module: str, filename: str) -> str:
    if module == "simulation":
        return "Four-area model parameters" if "four_area" in filename else "Additional page-specific guidance"
    if module == "field-potential":
        return "Kernel workflow and data-source guidance"
    if module == "features":
        return "Data loading and parser workflow guidance"
    if module == "inference":
        return "Training workflow guidance" if "new_training" in filename else "Prediction workflow guidance"
    return "Additional page-specific guidance"


def parse_template_help() -> dict[str, dict[str, list[tuple[str, str]]]]:
    grouped = {module: OrderedDict() for module in MODULES}
    seen = {module: set() for module in MODULES}
    for path in sorted(TEMPLATES.glob("*.html")):
        module = next(
            (value for prefix, value in MODULE_BY_TEMPLATE_PREFIX.items() if path.name.startswith(prefix)),
            None,
        )
        if module not in grouped:
            continue
        parser = StaticHelpParser()
        parser.feed(path.read_text(encoding="utf-8"))
        for label, text in parser.results:
            identity = (label, text)
            if identity not in seen[module]:
                category = template_group(module, path.name)
                grouped[module].setdefault(category, []).append(identity)
                seen[module].add(identity)
    return grouped


def build_reference_groups(source: str) -> dict[str, OrderedDict[str, list[tuple[str, str]]]]:
    grouped = {module: OrderedDict() for module in MODULES}
    overview = merge_grouped(parse_dashboard_help(), parse_menu_choices(), parse_page_help(source))
    fields = parse_main_field_help(source)
    templates = parse_template_help()
    for module in MODULES:
        grouped[module]["Module overview and available workflows"] = overview[module]
        if module == "simulation":
            continue
        for key, label, text in fields[module]:
            grouped[module].setdefault(field_group(module, key), []).append((label, text))
        for category, entries in templates[module].items():
            grouped[module].setdefault(category, []).extend(entries)

    simulation_fields = {key: (label, text) for key, label, text in fields["simulation"]}
    control_keys = {
        "tstop", "dt", "local_num_threads", "sim_run_mode", "sim_numpy_seed",
        "grid_start", "grid_step", "grid_end", "sim_repetitions",
    }
    hagen_keys = {
        "x", "n_x", "c_m_x", "tau_m_x", "e_l_x", "model", "c_yx", "j_yx",
        "delay_yx", "tau_syn_yx", "n_ext", "nu_ext", "j_ext",
    }
    cavallari_keys = {
        "x", "n_x", "c_m_x", "tau_m_x", "e_l_x", "model", "p", "extent", "ou_sigma",
        "ou_tau", "v_th_x", "v_reset_x", "t_ref_x", "g_l_x", "e_ex_x", "e_in_x",
        "tau_rise_ampa_x", "tau_decay_ampa_x", "tau_rise_gaba_a_x", "tau_decay_gaba_a_x", "i_e_x",
    }
    four_area_keys = hagen_keys | {"areas", "four_area_local_editor"}
    cavallari_fields = {
        key: simulation_fields[key]
        for key in simulation_fields
        if key in cavallari_keys
    }
    for key, text in parse_js_object(source, "cavallariHelpByKey"):
        cavallari_fields[key] = (humanize_key(key), text)
    four_area_fields = {
        key: simulation_fields[key]
        for key in simulation_fields
        if key in four_area_keys
    }
    for key, text in parse_js_object(source, "fourAreaHelpByKey"):
        four_area_fields[key] = (humanize_key(key), text)

    grouped["simulation"]["Simulation and parameter-sweep controls"] = [
        simulation_fields[key] for key in simulation_fields if key in control_keys
    ]
    grouped["simulation"]["Hagen model parameters"] = [
        simulation_fields[key] for key in simulation_fields if key in hagen_keys
    ]
    grouped["simulation"]["Cavallari model parameters"] = list(cavallari_fields.values())
    grouped["simulation"]["Four-area model parameters"] = list(four_area_fields.values())
    for category, entries in templates["simulation"].items():
        grouped["simulation"].setdefault(category, []).extend(entries)
    return grouped


def merge_grouped(*sources: dict[str, list[tuple[str, str]]]) -> dict[str, list[tuple[str, str]]]:
    result = {module: [] for module in MODULES}
    for source in sources:
        for module, entries in source.items():
            result[module].extend(entries)
    return result


def render_rows(entries: list[tuple[str, str]]) -> str:
    return "\n".join(
        f"""                <tr class="tooltip-entry">
                  <th scope="row"><code>{html.escape(label)}</code></th>
                  <td>{html.escape(text)}</td>
                </tr>"""
        for label, text in entries
    )


def render_group(title: str, entries: list[tuple[str, str]], open_group: bool = False) -> str:
    if not entries:
        return ""
    open_attribute = " open" if open_group else ""
    entry_label = "entry" if len(entries) == 1 else "entries"
    return f"""          <details class="config-group"{open_attribute}>
            <summary><span>{html.escape(title)}</span><span class="entry-count">{len(entries)} {entry_label}</span></summary>
            <div class="tooltip-table-wrap">
            <table class="tooltip-table">
              <thead><tr><th>Page, option, or parameter</th><th>Detailed information</th></tr></thead>
              <tbody>
{render_rows(entries)}
              </tbody>
            </table>
            </div>
          </details>
"""


def sidebar(active: bool = True) -> str:
    active_class = " active" if active else ""
    return f"""              <li class="subnav-group">WebUI Reference</li>
              <li><a class="subnav2{active_class}" href="./webui-information-reference.html">Configuration Reference</a></li>
              <li class="subnav-group">Simulation Pipeline</li>
              <li><a class="subnav2" href="./simulate-lif-network.html">1. Simulate a LIF Network</a></li>
              <li><a class="subnav2" href="./compute-field-potentials.html">2. Compute Field Potentials</a></li>
              <li><a class="subnav2" href="./extract-features.html">3. Feature Extraction</a></li>
              <li><a class="subnav2" href="./train-inverse-model.html">4. Inverse Model Training</a></li>
              <li><a class="subnav2" href="./compute-predictions.html">5. Compute Predictions</a></li>
              <li class="subnav-group">Empirical Dataset Analysis</li>
              <li><a class="subnav2" href="./empirical-data-processing.html">1. Load Empirical Data, Extract Features, and Compute Predictions</a></li>
              <li><a class="subnav2" href="./plot-empirical-results.html">2. Plot Empirical Results</a></li>"""


def generate() -> None:
    js_source = TOOLTIP_JS.read_text(encoding="utf-8")
    reference_groups = build_reference_groups(js_source)

    module_sections = []
    module_cards = []
    right_nav = []
    for module, title in MODULES.items():
        entry_count = sum(len(entries) for entries in reference_groups[module].values())
        right_nav.append(f'        <li><a href="#{module}">{html.escape(title)}</a></li>')
        module_cards.append(
            f"""            <a class="module-reference-card" href="#{module}">
              <strong>{html.escape(title)}</strong>
              <span>{html.escape(MODULE_DESCRIPTIONS[module])}</span>
              <small>{len(reference_groups[module])} categories · {entry_count} entries</small>
            </a>"""
        )
        rendered_groups = [
            render_group(group_title, entries, index == 0)
            for index, (group_title, entries) in enumerate(reference_groups[module].items())
        ]
        module_sections.append(
            f"""        <section id="{module}" class="tooltip-module">
          <div class="module-heading">
            <div>
              <span class="module-kicker">WebUI module</span>
              <h2>{html.escape(title)}</h2>
              <p>{html.escape(MODULE_DESCRIPTIONS[module])}</p>
            </div>
            <span class="module-count">{entry_count} entries</span>
          </div>
{chr(10).join(rendered_groups)}
        </section>"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ncpi Docs | WebUI Configuration Reference</title>
  <link rel="icon" type="image/png" href="https://raw.githubusercontent.com/necolab-ugr/ncpi/main/img/ncpi_logo.png">
  <link rel="stylesheet" href="../../docs_style.css">
  <style>
    .reference-overview {{ background: linear-gradient(135deg, #fff, rgba(0, 137, 199, 0.06)); }}
    .tooltip-search {{ width: 100%; padding: 0.78rem 0.9rem; border: 1px solid var(--line-strong); border-radius: 10px; font: inherit; background: #fff; }}
    .module-reference-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.75rem; margin-top: 1.1rem; }}
    .module-reference-card {{ display: grid; gap: 0.32rem; padding: 0.9rem; border: 1px solid rgba(0, 137, 199, 0.22); border-radius: 12px; background: #fff; color: var(--ink); }}
    .module-reference-card:hover {{ border-color: var(--line-strong); box-shadow: var(--shadow-soft); }}
    .module-reference-card span {{ color: var(--muted); font-size: 0.88rem; }}
    .module-reference-card small {{ color: var(--accent); font-family: "IBM Plex Mono", monospace; }}
    .tooltip-module {{ border-top: 4px solid var(--module-accent, var(--accent)); scroll-margin-top: 1rem; }}
    #simulation {{ --module-accent: #0089c7; }}
    #field-potential {{ --module-accent: #00a795; }}
    #features {{ --module-accent: #1f4dbf; }}
    #inference {{ --module-accent: #d9593b; }}
    #analysis {{ --module-accent: #7750b8; }}
    .module-heading {{ display: flex; justify-content: space-between; gap: 1rem; align-items: flex-start; margin-bottom: 1rem; }}
    .module-heading h2 {{ margin: 0.12rem 0 0.3rem; }}
    .module-heading p {{ margin: 0; }}
    .module-kicker {{ color: var(--module-accent, var(--accent)); font-family: "IBM Plex Mono", monospace; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }}
    .module-count, .entry-count {{ white-space: nowrap; border-radius: 999px; background: rgba(0, 137, 199, 0.09); color: var(--module-accent, var(--accent)); padding: 0.25rem 0.55rem; font-size: 0.72rem; font-weight: 700; }}
    .config-group {{ margin: 0.72rem 0; border: 1px solid rgba(0, 137, 199, 0.2); border-radius: 12px; overflow: hidden; background: #fff; }}
    .config-group summary {{ display: flex; justify-content: space-between; align-items: center; gap: 1rem; cursor: pointer; padding: 0.78rem 0.9rem; color: #0d3f63; font-family: "Sora", sans-serif; font-weight: 700; background: rgba(0, 137, 199, 0.045); }}
    .config-group[open] summary {{ border-bottom: 1px solid rgba(0, 137, 199, 0.18); }}
    .tooltip-table-wrap {{ overflow-x: auto; }}
    .tooltip-table {{ display: table; border-collapse: collapse; }}
    .tooltip-table th, .tooltip-table td {{ padding: 0.72rem 0.8rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    .tooltip-table tr:last-child th, .tooltip-table tr:last-child td {{ border-bottom: 0; }}
    .tooltip-table th:first-child {{ min-width: 14rem; width: 27%; }}
    .tooltip-table td {{ min-width: 28rem; }}
    .tooltip-empty {{ display: none; color: var(--muted); font-style: italic; }}
    @media (max-width: 760px) {{ .module-reference-grid {{ grid-template-columns: 1fr; }} .module-heading {{ display: block; }} .module-count {{ display: inline-block; margin-top: 0.65rem; }} }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar-left">
      <div class="brand">
        <a class="brand-home" href="../../index.html"><img src="https://raw.githubusercontent.com/necolab-ugr/ncpi/main/img/ncpi_logo.png" alt="ncpi logo"></a>
        <div class="title"><a href="../../index.html">Documentation</a></div>
      </div>
      <ul class="left-nav">
        <li><a class="github-link" href="https://github.com/necolab-ugr/ncpi"><img class="github-mark" src="https://github.githubassets.com/favicons/favicon.svg" alt=""/>GitHub Repository</a></li>
        <li><a href="../../index.html">Home</a></li>
        <li><a href="../../installation.html">Installation</a></li>
        <li><a class="active" href="../../tutorials.html">Tutorials</a></li>
        <li>
          <details class="left-nav-collapse" open>
            <summary class="subnav active">WebUI</summary>
            <ul class="left-nav nested-nav">
{sidebar()}
            </ul>
          </details>
        </li>
        <li>
          <details class="left-nav-collapse">
            <summary class="subnav">Jupyter</summary>
            <ul class="left-nav nested-nav">
              <li class="subnav-group">Simulation Pipeline</li>
              <li><a class="subnav2" href="../jupyter/simulate-lif-network.html">1. Simulate a LIF Network</a></li>
              <li><a class="subnav2" href="../jupyter/compute-field-potentials.html">2. Compute Field Potentials</a></li>
              <li><a class="subnav2" href="../jupyter/extract-features.html">3. Feature Extraction</a></li>
              <li><a class="subnav2" href="../jupyter/training-predictions.html">4 - Training an Inverse Model and Computing Predictions</a></li>
              <li class="subnav-group">Empirical Dataset Analysis</li>
              <li><a class="subnav2" href="../jupyter/empirical-data-processing.html">1. Load Empirical Data, Extract Features, and Compute Predictions</a></li>
            </ul>
          </details>
        </li>
        <li><a href="../../api.html">API</a></li>
        <li><a href="../../faq.html">FAQ</a></li>
        <li><a href="../../contributing.html">Contributing</a></li>
        <li><a href="../../citation.html">Citation</a></li>
        <li><a href="../../credits.html">Credits</a></li>
      </ul>
    </aside>

    <main class="content">
      <article class="doc">
        <section id="overview" class="reference-overview">
          <h1>WebUI Configuration Reference</h1>
          <p><span class="pill">Modules and Parameters</span></p>
          <p>
            This reference summarizes the purpose, workflows, and configurable parameters of every WebUI
            module. Information is organized by module, task, feature method, and simulation model so related
            settings can be reviewed together. Generic fallback messages for controls without explicit help
            are not repeated here.
          </p>
          <label for="tooltip-search"><strong>Search the configuration reference</strong></label>
          <input id="tooltip-search" class="tooltip-search" type="search" placeholder="Search modules, categories, parameters, and descriptions">
          <p id="tooltip-empty" class="tooltip-empty">No configuration information matches this search.</p>
          <div class="module-reference-grid">
{chr(10).join(module_cards)}
          </div>
        </section>

{chr(10).join(module_sections)}

        <p><a class="link" href="../webui-index.html">Back to WebUI Tutorials Index</a></p>
      </article>
    </main>

    <aside class="sidebar-right">
      <h2 class="right-title">On This Page</h2>
      <ul class="right-nav">
        <li><a class="active" href="#overview">Overview</a></li>
{chr(10).join(right_nav)}
      </ul>
    </aside>
  </div>
  <script src="../../nav-submenu-link.js"></script>
  <script>
    const search = document.getElementById("tooltip-search");
    const entries = Array.from(document.querySelectorAll(".tooltip-entry"));
    const tables = Array.from(document.querySelectorAll(".tooltip-table"));
    const groups = Array.from(document.querySelectorAll(".config-group"));
    const modules = Array.from(document.querySelectorAll(".tooltip-module"));
    const cards = Array.from(document.querySelectorAll(".module-reference-card"));
    const empty = document.getElementById("tooltip-empty");
    search.addEventListener("input", () => {{
      const query = search.value.trim().toLowerCase();
      modules.forEach((module) => {{
        const moduleMatches = Boolean(query) && module.querySelector(".module-heading").textContent.toLowerCase().includes(query);
        module.querySelectorAll(".config-group").forEach((group) => {{
          const groupMatches = moduleMatches || (Boolean(query) && group.querySelector("summary").textContent.toLowerCase().includes(query));
          group.querySelectorAll(".tooltip-entry").forEach((entry) => {{
            entry.hidden = Boolean(query) && !groupMatches && !entry.textContent.toLowerCase().includes(query);
          }});
          const matches = Boolean(group.querySelector(".tooltip-entry:not([hidden])"));
          group.hidden = !matches;
          if (query && matches) group.open = true;
        }});
        module.hidden = !module.querySelector(".config-group:not([hidden])");
      }});
      tables.forEach((table) => {{ table.closest(".tooltip-table-wrap").hidden = !table.querySelector(".tooltip-entry:not([hidden])"); }});
      cards.forEach((card) => {{
        const target = document.querySelector(card.getAttribute("href"));
        card.hidden = Boolean(query) && (!target || target.hidden);
      }});
      empty.style.display = entries.some((entry) => !entry.hidden) ? "none" : "block";
    }});
  </script>
</body>
</html>
"""
    OUTPUT.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    generate()
