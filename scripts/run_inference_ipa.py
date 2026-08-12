"""Run AtlasFold monomer IPA inference."""

import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold_IPA
from atlasfold.pretrained import load_model as load_pretrained_model
from atlasfold.runner_ipa import IPAFoldingInput, IPAFoldingOutput, IPAFoldingRunner

logger = logging.getLogger("atlasfold.ipa")


def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(name)s | %(message)s", datefmt="%y/%m/%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AtlasFold monomer IPA inference on a FASTA file."
    )
    parser.add_argument("-i", "--input-fasta", type=Path, required=True)
    parser.add_argument("-o", "--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--num-recycles", type=int, default=4)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument(
        "--recycle-early-stop-tolerance",
        type=float,
        default=0.0,
        help="Stop when pseudo-beta distance-matrix RMS change is below this value.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=[1])
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--max-tokens-per-batch", type=int, default=1024)
    parser.add_argument("--length-buckets", type=int, nargs="+", default=None)
    parser.add_argument("--format", choices=["cif", "pdb"], default="cif")
    parser.add_argument("--save-confidence-arrays", action="store_true")
    parser.add_argument("--save-distogram", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_inputs(path: Path) -> list[IPAFoldingInput]:
    if not path.exists():
        raise FileNotFoundError(f"Input FASTA does not exist: {path}")
    inputs = [
        IPAFoldingInput(header.split()[0].strip(), sequence)
        for header, sequence in read_fasta(path)
    ]
    if not inputs:
        raise ValueError(f"No sequences found in FASTA file: {path}")
    names = [item.name for item in inputs]
    if len(names) != len(set(names)):
        raise ValueError("FASTA target names must be unique.")
    return inputs


def load_model(args: argparse.Namespace) -> AtlasFold_IPA:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(
        "atlasfold-ipa",
        device=device,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
    )
    if not isinstance(model, AtlasFold_IPA):
        raise TypeError(f"Expected AtlasFold_IPA, got {type(model)!r}.")
    return model


def _log_recycle(name: str, seed: int, recycle: int, record: dict) -> None:
    tol = record["tol"]
    tol_text = "n/a" if tol is None else f"{tol:.3f}"
    logger.info(
        "%s seed=%d recycle=%d tol=%s pLDDT=%.3f pTM=%.3f avg_PAE=%.3f",
        name,
        seed,
        recycle,
        tol_text,
        record["avg_plddt"],
        record["ptm"],
        record["avg_pae"],
    )


def write_outputs(
    out_dir: Path,
    output: IPAFoldingOutput,
    *,
    format: str,
    save_confidence_arrays: bool,
    save_distogram: bool,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "done.txt"
    if done_path.exists():
        done_path.unlink()

    records = []
    for seed, prediction in sorted(output.outputs.items()):
        prefix = f"{output.name}_seed-{seed}"
        structure = prediction.to_mmcif() if format == "cif" else prediction.to_pdb()
        confidence = prediction.confidence_scores
        (out_dir / f"{prefix}_model.{format}").write_text(structure)
        (out_dir / f"{prefix}_confidence.json").write_text(
            json.dumps(confidence, indent=2)
        )
        (out_dir / f"{prefix}_recycles.json").write_text(
            json.dumps(output.recycle_metrics[seed], indent=2)
        )
        if save_confidence_arrays:
            np.savez(
                out_dir / f"{prefix}_confidence.npz",
                plddt=prediction.plddt,
                pae=prediction.pae,
            )
        history = output.recycle_metrics[seed]
        records.append(
            {
                "seed": seed,
                "avg_plddt": prediction.avg_plddt,
                "ptm": prediction.ptm,
                "num_recycles": len(history) - 1,
                "final_tol": history[-1]["tol"],
            }
        )
        if seed == output.best_seed:
            (out_dir / f"{output.name}_ranked_model.{format}").write_text(structure)
            (out_dir / f"{output.name}_ranked_confidence.json").write_text(
                json.dumps(confidence, indent=2)
            )

    if save_distogram:
        if not output.distogram_logits or output.distogram_boundaries is None:
            raise ValueError(f"No distograms are available for {output.name!r}.")
        for seed, logits in sorted(output.distogram_logits.items()):
            np.savez(
                out_dir / f"{output.name}_seed-{seed}_distogram.npz",
                logits=logits,
                boundaries=output.distogram_boundaries,
            )

    with (out_dir / f"{output.name}_summary.csv").open("w") as handle:
        handle.write("seed,avg_plddt,ptm,num_recycles,final_tol\n")
        for record in records:
            final_tol = (
                "" if record["final_tol"] is None else f"{record['final_tol']:.3f}"
            )
            handle.write(
                f"{record['seed']},{record['avg_plddt']:.3f},{record['ptm']:.3f},"
                f"{record['num_recycles']},{final_tol}\n"
            )
    done_path.touch()
    return next(record for record in records if record["seed"] == output.best_seed)


def run(args: argparse.Namespace) -> None:
    setup_logging()
    inputs = load_inputs(args.input_fasta)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        inputs = [
            item for item in inputs if not (out_dir / item.name / "done.txt").exists()
        ]
    if not inputs:
        logger.info("All targets are complete. Nothing to do.")
        return

    model = load_model(args)
    if args.no_kernel:
        model.set_forward_flags(use_cuequiv_kernels=False)
    runner = IPAFoldingRunner(model)

    start = timer()
    completed = 0
    for outputs in runner.fold_iter_batch(
        inputs,
        seeds=args.seed,
        num_recycles=args.num_recycles,
        mlm_prob=args.mlm_prob,
        recycle_early_stop_tolerance=args.recycle_early_stop_tolerance,
        length_buckets=args.length_buckets,
        max_tokens_per_batch=args.max_tokens_per_batch,
        return_distogram=args.save_distogram,
        recycle_callback=_log_recycle,
    ):
        for output in outputs:
            best = write_outputs(
                out_dir / output.name,
                output,
                format=args.format,
                save_confidence_arrays=args.save_confidence_arrays,
                save_distogram=args.save_distogram,
            )
            completed += 1
            logger.info(
                "Completed %s: pLDDT=%.3f pTM=%.3f recycles=%d (%d/%d)",
                output.name,
                best["avg_plddt"],
                best["ptm"],
                best["num_recycles"],
                completed,
                len(inputs),
            )
    logger.info("Finished %d target(s) in %.1fs.", completed, timer() - start)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    run(create_parser().parse_args())
