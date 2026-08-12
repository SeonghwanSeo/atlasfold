"""Command-line inference pipeline for AtlasFold multimer IPA models."""

import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np


def create_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run AtlasFold multimer IPA inference on colon-separated FASTA.",
    )
    required = parser.add_argument_group("required arguments")
    inference = parser.add_argument_group("inference options")
    runtime = parser.add_argument_group("runtime and batching options")
    output = parser.add_argument_group("output options")

    required.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help=(
            "Input FASTA file. Each record is one complex, with ':' separating chains."
        ),
    )
    required.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )
    required.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local multimer IPA model weights.",
    )

    inference.add_argument(
        "--num-recycles",
        type=int,
        default=10,
        help="Maximum number of recycling iterations.",
    )
    inference.add_argument(
        "--mlm-prob",
        type=float,
        default=0.20,
        help="LM masking probability used during recycling.",
    )
    inference.add_argument(
        "--recycle-early-stop-tolerance",
        type=float,
        default=0.0,
        help="Stop when pseudo-beta distance-matrix RMS change is below this value.",
    )
    inference.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Random seed(s) for inference. Example: --seed 1 2 3.",
    )
    inference.add_argument(
        "--print-recycle-metrics",
        action="store_true",
        help="Print confidence and convergence metrics after every recycle.",
    )

    runtime.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for AtlasLM weights.",
    )
    runtime.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    runtime.add_argument(
        "--no-kernel",
        action="store_true",
        help="Disable cuequivariance kernels.",
    )
    runtime.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help="Maximum bucketed residue tokens per model call; 0 disables batching.",
    )
    runtime.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit residue buckets.",
    )

    output.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for seeded and ranked outputs.",
    )
    output.add_argument(
        "--save-confidence-arrays",
        action="store_true",
        help="Save raw pLDDT and PAE arrays for each seed as NPZ files.",
    )
    output.add_argument(
        "--save-distogram",
        action="store_true",
        help="Save raw distogram logits and boundaries for each seed as NPZ files.",
    )
    output.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def run(args: argparse.Namespace) -> None:
    if args.num_recycles < 0:
        raise ValueError(f"num_recycles must be non-negative, got {args.num_recycles}.")
    if not 0.0 < args.mlm_prob <= 1.0:
        raise ValueError(
            f"mlm_prob must be greater than 0 and at most 1, got {args.mlm_prob}."
        )
    if args.recycle_early_stop_tolerance < 0:
        raise ValueError(
            "recycle_early_stop_tolerance must be non-negative, "
            f"got {args.recycle_early_stop_tolerance}."
        )
    if args.max_tokens_per_batch < 0:
        raise ValueError(
            f"max_tokens_per_batch must be non-negative, got {args.max_tokens_per_batch}."
        )
    if args.length_buckets is not None and any(
        bucket <= 0 for bucket in args.length_buckets
    ):
        raise ValueError(
            "length_buckets must contain only positive values, "
            f"got {args.length_buckets}."
        )

    import torch

    from atlasfold.data.fasta import read_fasta
    from atlasfold.model import AtlasFoldMultimer_IPA
    from atlasfold.pretrained import load_model as load_pretrained_model
    from atlasfold.runner_multimer_ipa import (
        MultimerIPAFoldingInput,
        MultimerIPAFoldingOutput,
        MultimerIPAFoldingRunner,
    )

    def setup_logger() -> logging.Logger:
        logger = logging.getLogger("atlasfold.multimer_ipa")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(name)s | %(message)s",
                    datefmt="%y/%m/%d %H:%M:%S",
                )
            )
            logger.addHandler(handler)
        return logger

    def load_inputs(path: Path) -> list[MultimerIPAFoldingInput]:
        if not path.exists():
            raise FileNotFoundError(f"Input FASTA does not exist: {path}")
        inputs = []
        for header, sequence in read_fasta(path):
            header_fields = header.split()
            if not header_fields:
                raise ValueError("FASTA target has an empty header.")
            name = header_fields[0]
            if name in {".", ".."} or "/" in name or "\\" in name:
                raise ValueError(
                    f"FASTA target name must not contain path components: {name!r}."
                )
            chains = sequence.split(":")
            if not chains or any(not chain.strip() for chain in chains):
                raise ValueError(f"FASTA target {name!r} contains an empty chain.")
            inputs.append(MultimerIPAFoldingInput(name, chains))
        if not inputs:
            raise ValueError(f"No sequences found in FASTA file: {path}")
        names = [item.name for item in inputs]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                "FASTA target names must be unique after normalization. "
                f"Duplicates: {duplicates}"
            )
        return inputs

    def load_model() -> AtlasFoldMultimer_IPA:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pretrained_model(
            "atlasfold-multimer-ipa",
            device=device,
            cache_dir=args.cache_dir,
            model_path=args.model_path,
        )
        if not isinstance(model, AtlasFoldMultimer_IPA):
            raise TypeError(f"Expected AtlasFoldMultimer_IPA, got {type(model)!r}.")
        return model

    def print_recycle_metrics(name: str, seed: int, recycle: int, record: dict) -> None:
        tol = record["tol"]
        message = (
            f"{name}_seed_{seed:03d} recycle={recycle} "
            f"pLDDT={record['plddt']:.1f} pTM={record['ptm']:.3f}"
        )
        if tol is not None:
            message += f" tol={tol:.2f}"
        logger.info(message)

    def write_outputs(out_dir: Path, output: MultimerIPAFoldingOutput) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        done_path = out_dir / "done.txt"
        if done_path.exists():
            done_path.unlink()

        records = []
        for seed, prediction in sorted(output.outputs.items()):
            prefix = f"{output.name}_seed-{seed}"
            structure = (
                prediction.to_mmcif() if args.format == "cif" else prediction.to_pdb()
            )
            confidence = prediction.confidence_scores
            (out_dir / f"{prefix}_model.{args.format}").write_text(structure)
            (out_dir / f"{prefix}_confidence.json").write_text(
                json.dumps(confidence, indent=2)
            )
            if args.save_confidence_arrays:
                np.savez(
                    out_dir / f"{prefix}_confidence.npz",
                    plddt=prediction.plddt,
                    pae=prediction.pae,
                )
            records.append(
                {
                    "seed": seed,
                    "ranking_score": prediction.ranking_score,
                    "avg_plddt": prediction.avg_plddt,
                    "ptm": prediction.ptm,
                    "iptm": prediction.iptm,
                    "num_recycles": output.recycle_counts[seed],
                }
            )
            if seed == output.best_seed:
                (out_dir / f"{output.name}_ranked_model.{args.format}").write_text(
                    structure
                )
                (out_dir / f"{output.name}_ranked_confidence.json").write_text(
                    json.dumps(confidence, indent=2)
                )

        if args.save_distogram:
            if not output.distogram_logits or output.distogram_boundaries is None:
                raise ValueError(f"No distograms are available for {output.name!r}.")
            for seed, logits in sorted(output.distogram_logits.items()):
                np.savez(
                    out_dir / f"{output.name}_seed-{seed}_distogram.npz",
                    logits=logits,
                    boundaries=output.distogram_boundaries,
                )

        with (out_dir / f"{output.name}_summary.csv").open("w") as handle:
            handle.write("seed,ranking_score,avg_plddt,ptm,iptm,num_recycles\n")
            for record in records:
                handle.write(
                    f"{record['seed']},{record['ranking_score']:.3f},"
                    f"{record['avg_plddt']:.3f},{record['ptm']:.3f},"
                    f"{record['iptm']:.3f},{record['num_recycles']}\n"
                )
        done_path.touch()
        return next(record for record in records if record["seed"] == output.best_seed)

    logger = setup_logger()
    torch.set_float32_matmul_precision("highest")
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

    model = load_model()
    if args.no_kernel:
        model.set_forward_flags(use_cuequiv_kernels=False)
    runner = MultimerIPAFoldingRunner(model)

    max_tokens_per_batch = args.max_tokens_per_batch
    if (
        args.recycle_early_stop_tolerance > 0 or args.print_recycle_metrics
    ) and max_tokens_per_batch != 0:
        logger.warning(
            "Disabling batching because recycle early stopping or metric printing "
            "is enabled (max_tokens_per_batch: %d -> 0).",
            max_tokens_per_batch,
        )
        max_tokens_per_batch = 0

    start = timer()
    batch_start = timer()
    completed = 0
    for outputs in runner.fold_iter_batch(
        inputs,
        seeds=args.seed,
        num_recycles=args.num_recycles,
        mlm_prob=args.mlm_prob,
        recycle_early_stop_tolerance=args.recycle_early_stop_tolerance,
        length_buckets=args.length_buckets,
        max_tokens_per_batch=max_tokens_per_batch,
        return_distogram=args.save_distogram,
        recycle_callback=(print_recycle_metrics if args.print_recycle_metrics else None),
    ):
        batch_elapsed = timer() - batch_start
        time_per_target = batch_elapsed / len(outputs)
        batch_start = timer()
        for output in outputs:
            best = write_outputs(
                out_dir / output.name,
                output,
            )
            completed += 1
            logger.info(
                "Completed %s (length=%d): pLDDT=%.3f pTM=%.3f ipTM=%.3f "
                "recycles=%d time=%.2f (%d/%d)",
                output.name,
                output.length,
                best["avg_plddt"],
                best["ptm"],
                best["iptm"],
                best["num_recycles"],
                time_per_target,
                completed,
                len(inputs),
            )
    elapsed = timer() - start
    logger.info(
        "Finished %d target(s) in %.1fs (%.1fs per target).",
        completed,
        elapsed,
        elapsed / completed,
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
