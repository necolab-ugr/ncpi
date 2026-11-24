"""
Build a small subset of the developmental mouse LFP dataset (Chini et al., 2022) for quick tests.

The full dataset (Zenodo record 15382047, ~11 GB) is downloaded by LFP_developing_brain.py to
ncpi_data/empirical/development_EI_decorrelation. This script keeps a few recordings per age, crops
each one to its first seconds and writes them with the same folder layout and .mat structure, so
load_empirical_data() in LFP_developing_brain.py works unchanged on the subset.

Usage:
    python make_small_LFP_dataset.py --source <.../development_EI_decorrelation> --output <dir>
"""

import argparse
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio

DATASET_NAME = "development_EI_decorrelation"
LFP_SUBDIR = Path("baseline") / "LFP"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "ncpi_data" / "empirical" / DATASET_NAME
DEFAULT_OUTPUT = REPO_ROOT / "ncpi_data" / "empirical_small"

README_TEMPLATE = """# Developmental mouse LFP recordings: small subset

Small subset of the baseline LFP recordings from:

Chini, M., Pfeffer, T., & Hanganu-Opatz, I. (2022). An increase of inhibition drives the developmental
decorrelation of neural activity. eLife, 11, e78811. https://doi.org/10.7554/eLife.78811

Original data: https://gin.g-node.org/mchini/development_EI_decorrelation
Full dataset on Zenodo: https://doi.org/10.5281/zenodo.15382047

Subset contents:
- {n_files} recordings from baseline/LFP: {per_age} per postnatal age (P{age_min} to P{age_max})
- each recording cropped to its first {seconds:g} s (16 channels, {fs:g} Hz)
- same .mat structure as the original files (fields: LFP, age, fs, downsampling_factor, filter)

Files (original file name: age, samples):
{file_list}

Generated with examples/LFP_developing_brain/make_small_LFP_dataset.py from the ncpi repository
(https://github.com/necolab-ugr/ncpi). Released under CC0 1.0, as the original dataset.
"""


def _file_number(path):
    match = re.match(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _load_lfp(path):
    return sio.loadmat(path, struct_as_record=False, squeeze_me=True)["LFP"]


def select_files(lfp_dir, per_age):
    """Pick the first `per_age` recordings (by file number) for each age."""
    by_age = defaultdict(list)
    for path in sorted(lfp_dir.glob("*.mat"), key=_file_number):
        by_age[int(_load_lfp(path).age)].append(path)
    return {age: paths[:per_age] for age, paths in sorted(by_age.items())}


def crop_recording(src, dst, seconds):
    """Save the first `seconds` of a recording keeping the original struct fields."""
    lfp = _load_lfp(src)
    n_samples = int(round(seconds * float(lfp.fs)))
    data = np.asarray(lfp.LFP)
    if data.shape[1] < n_samples:
        raise ValueError(f"{src.name} is shorter than {seconds} s.")
    struct = {name: getattr(lfp, name) for name in lfp._fieldnames}
    struct["LFP"] = data[:, :n_samples]
    sio.savemat(dst, {"LFP": struct}, do_compression=True)
    return int(lfp.age), n_samples, float(lfp.fs)


def build_subset(source, output, per_age, seconds):
    source_lfp = source / LFP_SUBDIR
    if not source_lfp.is_dir():
        raise FileNotFoundError(f"LFP folder not found: {source_lfp}")

    subset_root = output / DATASET_NAME
    if subset_root.exists():
        shutil.rmtree(subset_root)
    out_lfp = subset_root / LFP_SUBDIR
    out_lfp.mkdir(parents=True)

    rows = []
    for paths in select_files(source_lfp, per_age).values():
        for path in paths:
            age_saved, n_samples, fs = crop_recording(path, out_lfp / path.name, seconds)
            rows.append((path.name, age_saved, n_samples, fs))
            print(f"P{age_saved}: {path.name} -> {n_samples} samples")

    if (source / "LICENSE").is_file():
        shutil.copy2(source / "LICENSE", subset_root / "LICENSE")
    ages = [age for _, age, _, _ in rows]
    (subset_root / "README.md").write_text(
        README_TEMPLATE.format(
            n_files=len(rows),
            per_age=per_age,
            age_min=min(ages),
            age_max=max(ages),
            seconds=seconds,
            fs=rows[0][3],
            file_list="\n".join(f"- {name}: P{age}, {n} samples" for name, age, n, _ in rows),
        )
    )

    archive = output / f"{DATASET_NAME}_small.tar.xz"
    with tarfile.open(archive, "w:xz") as tar:
        tar.add(subset_root, arcname=DATASET_NAME)
    size_mb = archive.stat().st_size / 1e6
    print(f"\n{len(rows)} recordings written to {subset_root}")
    print(f"Archive: {archive} ({size_mb:.1f} MB)")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Full dataset folder.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder for the subset.")
    parser.add_argument("--per-age", type=int, default=2, help="Recordings kept per postnatal age.")
    parser.add_argument("--seconds", type=float, default=120.0, help="Duration kept from each recording (s).")
    args = parser.parse_args()
    build_subset(args.source, args.output, args.per_age, args.seconds)


if __name__ == "__main__":
    main()
