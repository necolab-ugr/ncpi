import argparse
import csv
import os
import pickle
import re
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BATCH_DIR_PATTERN = re.compile(r"batch_(\d{4,})$")
SIM_DATA_METHODS = ("proxy", "specparam", "catch22")
DEFAULT_SCRATCH_ROOT = "/SCRATCH/TIC117/pablomc/Cavallari_model_simulations"
DEFAULT_INPUT_ROOT = os.path.join(DEFAULT_SCRATCH_ROOT, "simulation_output")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_INPUT_ROOT, "merged")
DEFAULT_VALID_SAMPLES_PLOT = "valid_parameter_samples.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge outputs produced by multiple batch_XXXX runs of "
            "massive_model_simulation.py into a single folder."
        )
    )
    parser.add_argument("--input-root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_paths(input_root: str, output_dir: Optional[str]) -> Tuple[str, str]:
    resolved_input = os.path.abspath(os.path.expanduser(os.path.expandvars(input_root)))
    resolved_output = output_dir or DEFAULT_OUTPUT_DIR
    resolved_output = os.path.abspath(os.path.expanduser(os.path.expandvars(resolved_output)))
    return resolved_input, resolved_output


def find_batch_dirs(input_root: str) -> List[Tuple[int, str]]:
    batch_dirs = []
    for entry in sorted(os.listdir(input_root)):
        path = os.path.join(input_root, entry)
        match = BATCH_DIR_PATTERN.fullmatch(entry)
        if not match or not os.path.isdir(path):
            continue
        batch_dirs.append((int(match.group(1)), path))
    if not batch_dirs:
        raise FileNotFoundError(f"No batch_XXXX folders found under {input_root}.")
    return batch_dirs


def ensure_output_structure(output_dir: str, overwrite: bool) -> None:
    if os.path.exists(output_dir) and not overwrite:
        if os.listdir(output_dir):
            raise FileExistsError(
                f"Output directory {output_dir} already exists and is not empty. "
                "Use --overwrite or choose another directory."
            )
    os.makedirs(output_dir, exist_ok=True)
    for method in SIM_DATA_METHODS:
        os.makedirs(os.path.join(output_dir, "data", method), exist_ok=True)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_csv_rows(rows: Sequence[Dict[str, object]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_pickle(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_pickle(payload, path: str) -> None:
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_optional_int(value) -> Optional[int]:
    if value in (None, "", "None"):
        return None
    return int(value)


def normalize_batch_id(value, fallback_batch_id: int) -> int:
    if value in (None, "", "None"):
        return fallback_batch_id
    return int(value)


def remap_valid_sample_indices(
    all_rows: List[Dict[str, object]],
    valid_rows: List[Dict[str, object]],
    valid_details: List[dict],
    merged_output_dir: str,
) -> None:
    index_map: Dict[Tuple[int, int], int] = {}

    for global_index, detail in enumerate(valid_details):
        batch_id = normalize_batch_id(detail.get("batch_id"), -1)
        old_sample_index = parse_optional_int(detail.get("sample_index"))
        if old_sample_index is None:
            continue

        detail["sample_index"] = global_index
        detail["data_dirs"] = {
            method: os.path.join(merged_output_dir, "data", method)
            for method in SIM_DATA_METHODS
        }
        index_map[(batch_id, old_sample_index)] = global_index

    for row in all_rows:
        if not parse_bool(row.get("valid")):
            row["sample_index"] = ""
            continue

        batch_id = normalize_batch_id(row.get("batch_id"), -1)
        old_sample_index = parse_optional_int(row.get("sample_index"))
        if old_sample_index is None:
            row["sample_index"] = ""
            continue

        row["sample_index"] = index_map.get((batch_id, old_sample_index), "")

    for row in valid_rows:
        batch_id = normalize_batch_id(row.get("batch_id"), -1)
        old_sample_index = parse_optional_int(row.get("sample_index"))
        if old_sample_index is None:
            row["sample_index"] = ""
            continue
        row["sample_index"] = index_map.get((batch_id, old_sample_index), "")


def merge_sim_data_for_method(batch_dirs: Sequence[Tuple[int, str]], method: str, output_dir: str) -> None:
    theta_chunks = []
    x_chunks = []
    theta_parameters = None

    for _, batch_dir in batch_dirs:
        method_dir = os.path.join(batch_dir, "data", method)
        theta_payload = load_pickle(os.path.join(method_dir, "sim_theta"))
        x_payload = load_pickle(os.path.join(method_dir, "sim_X"))

        if theta_payload is None or x_payload is None:
            continue

        current_parameters = list(theta_payload.get("parameters", []))
        if theta_parameters is None:
            theta_parameters = current_parameters
        elif current_parameters != theta_parameters:
            raise ValueError(
                f"Inconsistent theta parameter order for method {method}: "
                f"{current_parameters} != {theta_parameters}"
            )

        theta_data = np.asarray(theta_payload.get("data"))
        x_data = np.asarray(x_payload)
        if theta_data.shape[0] != x_data.shape[0]:
            raise ValueError(
                f"Mismatched sample count for method {method} in {batch_dir}: "
                f"theta has {theta_data.shape[0]}, X has {x_data.shape[0]}."
            )

        theta_chunks.append(theta_data)
        x_chunks.append(x_data)

    if not theta_chunks:
        return

    merged_theta = {
        "parameters": theta_parameters,
        "data": np.concatenate(theta_chunks, axis=0).astype(np.float32),
    }
    merged_x = np.concatenate(x_chunks, axis=0).astype(np.float32)

    method_output_dir = os.path.join(output_dir, "data", method)
    save_pickle(merged_theta, os.path.join(method_output_dir, "sim_theta"))
    save_pickle(merged_x, os.path.join(method_output_dir, "sim_X"))


def plot_valid_parameter_samples(valid_details: Sequence[dict], output_dir: str) -> None:
    if not valid_details:
        return

    parameter_names = list(valid_details[0].get("parameters", {}).keys())
    if not parameter_names:
        return

    sampled_values = np.array(
        [
            [detail["parameters"][name] for name in parameter_names]
            for detail in valid_details
        ],
        dtype=float,
    )

    n_params = len(parameter_names)
    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(2.0 * n_params, 2.0 * n_params),
        dpi=150,
        constrained_layout=True,
    )

    if n_params == 1:
        axes = np.array([[axes]])

    for row_idx in range(n_params):
        for col_idx in range(n_params):
            ax = axes[row_idx, col_idx]
            if row_idx == col_idx:
                ax.hist(sampled_values[:, col_idx], bins=20, color="0.75", edgecolor="0.35")
            elif row_idx > col_idx:
                ax.scatter(
                    sampled_values[:, col_idx],
                    sampled_values[:, row_idx],
                    s=8,
                    alpha=0.35,
                    color="C0",
                    linewidths=0.0,
                )
            else:
                ax.axis("off")
                continue

            if row_idx == n_params - 1:
                ax.set_xlabel(parameter_names[col_idx], fontsize=8)
            else:
                ax.set_xticklabels([])
            if col_idx == 0:
                ax.set_ylabel(parameter_names[row_idx], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)

    figure_path = os.path.join(output_dir, DEFAULT_VALID_SAMPLES_PLOT)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    input_root, output_dir = resolve_paths(args.input_root, args.output_dir)
    batch_dirs = find_batch_dirs(input_root)
    ensure_output_structure(output_dir, args.overwrite)

    merged_all_rows: List[Dict[str, object]] = []
    merged_valid_rows: List[Dict[str, object]] = []
    merged_valid_details: List[dict] = []

    for batch_id, batch_dir in batch_dirs:
        all_rows = load_csv_rows(os.path.join(batch_dir, "all_simulations_summary.csv"))
        valid_rows = load_csv_rows(os.path.join(batch_dir, "valid_samples_summary.csv"))
        valid_details = load_pickle(os.path.join(batch_dir, "valid_samples_details.pkl")) or []

        for row in all_rows:
            row["batch_id"] = normalize_batch_id(row.get("batch_id"), batch_id)
        for row in valid_rows:
            row["batch_id"] = normalize_batch_id(row.get("batch_id"), batch_id)
        for detail in valid_details:
            detail["batch_id"] = normalize_batch_id(detail.get("batch_id"), batch_id)

        merged_all_rows.extend(all_rows)
        merged_valid_rows.extend(valid_rows)
        merged_valid_details.extend(valid_details)

    remap_valid_sample_indices(
        merged_all_rows,
        merged_valid_rows,
        merged_valid_details,
        output_dir,
    )

    save_csv_rows(merged_all_rows, os.path.join(output_dir, "all_simulations_summary.csv"))
    save_csv_rows(merged_valid_rows, os.path.join(output_dir, "valid_samples_summary.csv"))
    save_pickle(merged_valid_details, os.path.join(output_dir, "valid_samples_details.pkl"))

    for method in SIM_DATA_METHODS:
        merge_sim_data_for_method(batch_dirs, method, output_dir)

    plot_valid_parameter_samples(merged_valid_details, output_dir)

    print(f"Merged {len(batch_dirs)} batches into {output_dir}")
    print(f"Total simulations: {len(merged_all_rows)}")
    print(f"Total valid samples: {len(merged_valid_details)}")


if __name__ == "__main__":
    main()
