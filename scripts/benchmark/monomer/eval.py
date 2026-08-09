import argparse
import concurrent.futures
import csv
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

# to download the docker image, refer to:
# https://git.scicore.unibas.ch/schwede/openstructure#docker
OST_COMPARE_STRUCTURE = r"""
command="compare-structures \
-m {model_file} \
-r {reference_file} \
--fault-tolerant \
--min-pep-length 4 \
--min-nuc-length 4 \
-o {output_path} \
--lddt --bb-lddt \
--ics --ips --rigid-scores --patch-scores --tm-score"

ost $command
"""


METRICS = [
    "lddt",
    "bb_lddt",
    "tm_score",
    "rmsd",
    "oligo_gdtts",
]
DEFAULT_SAMPLE_PATTERN = "{name}_seed-*_sample-*_model.cif"
SAMPLE_INDEX_RE = re.compile(r"seed-(\d+)_sample-(\d+)")
TARGET_METRIC_COLUMNS = (
    ("GDT-TS", "oligo_gdtts", 100.0),
    ("TMscore", "tm_score", None),
    ("LDDT", "lddt", None),
    ("LDDT-BB", "bb_lddt", None),
    ("RMSD", "rmsd", None),
)
SUMMARY_METRIC_LABELS = (
    ("tm_score", "TM score"),
    ("oligo_gdtts", "GDT-TS  "),
    ("lddt", "LDDT    "),
    ("bb_lddt", "LDDT-BB "),
    ("rmsd", "RMSD    "),
)
RAW_SAMPLE_FIELDNAMES = [
    "target",
    "eval_stem",
    "seed",
    "sample",
    "avg_plddt",
    "ptm",
    "confidence_score",
    *METRICS,
]


def evaluate_structure(name: str, pred: Path, reference: Path, out: Path) -> str | None:
    """Evaluate one predicted structure against one reference structure."""
    if out.exists():
        print(f"Skipping recomputation of {name} as protein json file already exists")
        return None

    completed = subprocess.run(
        OST_COMPARE_STRUCTURE.format(
            model_file=str(pred),
            reference_file=str(reference),
            output_path=str(out),
        ),
        shell=True,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not out.exists():
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or f"return code {completed.returncode}"
        return f"{name}: {message.splitlines()[-1]}"
    return None


def reference_path(data_dir: Path, name: str) -> Path | None:
    for suffix in (".cif", ".pdb"):
        path = data_dir / f"{name}{suffix}"
        if path.exists():
            return path
    return None


def prediction_paths(sample_dir: Path, name: str) -> list[Path]:
    patterns = [f"{{name}}/{DEFAULT_SAMPLE_PATTERN}"]
    preds: list[Path] = []
    seen = set()
    for pattern in patterns:
        for pred in sorted(sample_dir.glob(pattern.format(name=name))):
            if pred not in seen:
                preds.append(pred)
                seen.add(pred)
    return preds


def sample_indices(eval_file: Path) -> tuple[int | None, int | None]:
    match = SAMPLE_INDEX_RE.search(eval_file.stem)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


def confidence_json_path(sample_dir: Path, name: str, eval_file: Path) -> Path:
    prediction_stem = eval_file.stem.removesuffix("_model")
    return sample_dir / name / f"{prediction_stem}_confidence.json"


def confidence_top1(eval_datas: list[dict]) -> dict | None:
    candidates = [
        eval_data
        for eval_data in eval_datas
        if eval_data.get("_confidence_score") is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda eval_data: eval_data["_confidence_score"])


def raw_sample_row(eval_data: dict, name: str) -> dict:
    row = {
        "target": name,
        "eval_stem": eval_data["_eval_stem"],
        "seed": eval_data["_seed"],
        "sample": eval_data["_sample"],
        "avg_plddt": eval_data["_avg_plddt"],
        "ptm": eval_data["_ptm"],
        "confidence_score": eval_data["_confidence_score"],
    }
    for metric_name in METRICS:
        row[metric_name] = eval_data.get(metric_name)
    return row


def run_eval(args) -> None:
    data_names = {
        x.stem: x
        for x in args.data.iterdir()
        if x.is_file() and not x.name.startswith(".")
    }

    args.outdir.mkdir(parents=True, exist_ok=True)

    failures = []
    with concurrent.futures.ThreadPoolExecutor(args.max_workers) as executor:
        futures = []
        for name in sorted(data_names):
            ref_path = reference_path(args.data, name)
            if ref_path is None:
                print(f"Reference file for {name} does not exist, skipping")
                continue

            preds = prediction_paths(args.sample, name)
            if not preds:
                print(f"No prediction samples found for {name}, skipping")
                continue

            target_outdir = args.outdir / name
            target_outdir.mkdir(parents=True, exist_ok=True)

            for pred_path in preds:
                out_path = target_outdir / f"{pred_path.stem}.json"
                if out_path.exists():
                    continue
                eval_name = f"{name}/{pred_path.stem}"
                futures.append(
                    executor.submit(
                        evaluate_structure,
                        name=eval_name,
                        pred=pred_path,
                        reference=ref_path,
                        out=out_path,
                    )
                )

        with tqdm(total=len(futures), leave=False) as pbar:
            for future in concurrent.futures.as_completed(futures):
                failure = future.result()
                if failure is not None:
                    failures.append(failure)
                pbar.update(1)

    if failures:
        print("Evaluation failures:")
        for failure in failures[:20]:
            print(f"  - {failure}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more failures")


def compute_metrics(
    evals: Path, sample_dir: Path, name: str
) -> tuple[dict[str, dict[str, float | None]], list[dict]]:
    metrics: dict[str, list[float]] = {}
    eval_files = [
        evals / name / f"{pred_path.stem}.json"
        for pred_path in prediction_paths(sample_dir, name)
    ]
    eval_files = [eval_file for eval_file in eval_files if eval_file.exists()]
    if not eval_files:
        raise ValueError(f"No evaluation json files found for {name}")

    eval_datas = []
    for eval_file in eval_files:
        try:
            with eval_file.open("r") as f:
                eval_data = json.load(f)
            seed, sample = sample_indices(eval_file)
            conf_path = confidence_json_path(sample_dir, name, eval_file)
            if not conf_path.exists():
                raise FileNotFoundError(
                    f"Confidence json file {conf_path} does not exist"
                )
            with conf_path.open("r") as f:
                confidence_data = json.load(f)
            plddt = confidence_data["avg_plddt"]
            ptm = confidence_data["ptm"]
            if plddt is None:
                raise ValueError(f"avg_plddt is None in {conf_path}")
            if ptm is None:
                raise ValueError(f"ptm is None in {conf_path}")
        except Exception as e:
            print(f"Error processing {eval_file}: {e}")
            continue

        eval_data["_eval_stem"] = eval_file.stem
        eval_data["_seed"] = seed
        eval_data["_sample"] = sample
        eval_data["_confidence_score"] = plddt
        eval_data["_avg_plddt"] = plddt
        eval_data["_ptm"] = ptm

        eval_datas.append(eval_data)

    confidence_all_eval_data = confidence_top1(eval_datas)
    if confidence_all_eval_data is None:
        raise ValueError(f"No confidence json files found for {name}")

    for eval_data in eval_datas:
        for metric_name in METRICS:
            if metric_name in eval_data:
                metrics.setdefault(metric_name, []).append(eval_data[metric_name])

    raw_sample_rows = [raw_sample_row(eval_data, name) for eval_data in eval_datas]

    results = {}
    for metric_name, values in metrics.items():
        if not values:
            continue
        if metric_name not in confidence_all_eval_data:
            continue
        results[metric_name] = {
            "top_confidence": confidence_all_eval_data.get(metric_name),
            "n_samples": len(values),
            "avg_plddt": confidence_all_eval_data.get("_avg_plddt"),
            "ptm": confidence_all_eval_data.get("_ptm"),
        }
    return results, raw_sample_rows


def print_summary(rows: list[dict]) -> None:
    values_by_metric: dict[str, list[float]] = {}
    for row in rows:
        value = row.get("top_confidence")
        if value is None:
            continue
        values_by_metric.setdefault(row["metric"], []).append(value)

    print("Avg/Med")
    for metric_name, label in SUMMARY_METRIC_LABELS:
        values = values_by_metric.get(metric_name)
        if not values:
            continue
        avg = np.mean(values)
        med = np.median(values)
        print(f"{label}: {avg:0.3f} / {med:0.3f}")


def aggregate_eval(args) -> list[dict]:
    data_names = {
        x.stem: x
        for x in args.data.iterdir()
        if x.is_file() and not x.name.startswith(".")
    }
    print("Number of data", len(data_names))

    all_results = []
    all_sample_results = []
    for name in sorted(data_names):
        try:
            results, sample_results = compute_metrics(args.outdir, args.sample, name)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            continue

        all_sample_results.extend(sample_results)

        for metric_name, values in results.items():
            all_results.append(
                {
                    "target": name,
                    "metric": metric_name,
                    "top_confidence": values["top_confidence"],
                    "n_samples": values["n_samples"],
                    "avg_plddt": values["avg_plddt"],
                    "ptm": values["ptm"],
                }
            )

    print(
        "Successfully read and processed evaluation JSON files:",
        len(all_sample_results),
    )

    samples_path = args.outdir.parent / "results_samples.csv"
    with samples_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_SAMPLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_sample_results)

    if not all_results:
        print("No evaluation results found.")
        print(f"Wrote per-sample results to {samples_path}")
        return []

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the PDB mmcif directory",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="Run directory containing predictions/",
    )
    parser.add_argument("--max-workers", type=int, default=128)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip OpenStructure and rebuild CSV summaries from existing eval JSONs.",
    )
    args = parser.parse_args()

    args.data = Path(args.data_dir)
    args.sample = Path(args.sample_dir) / "predictions/"
    args.outdir = Path(args.sample_dir) / "eval/"
    if not args.data.exists():
        raise ValueError(f"Data directory {args.data} does not exist")
    if not args.sample.exists():
        raise ValueError(f"Sample directory {args.sample} does not exist")

    if not args.aggregate_only:
        run_eval(args)
    results = aggregate_eval(args)
    print_summary(results)
