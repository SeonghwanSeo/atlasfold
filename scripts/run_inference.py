import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold, SamplingConfig
from atlasfold.pretrained import load_model as load_pretrained_model
from atlasfold.runner import FoldingInput, FoldingOutput, FoldingRunner

logger = logging.getLogger("atlasfold.monomer")


def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AtlasFold batched inference on a FASTA file."
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
        help="Path to the input FASTA file.",
    )
    required.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )

    inference.add_argument(
        "--num-recycles",
        type=int,
        default=4,
        help="Number of recycling iterations.",
    )
    inference.add_argument(
        "--mlm-prob",
        type=float,
        default=0.15,
        help="LM masking probability used during recycling.",
    )
    inference.add_argument(
        "--stochastic",
        action="store_true",
        help="Increase sampling diversity across seeds.",
    )
    inference.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate per input sequence.",
    )
    inference.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Random seed(s) for inference. Example: --seed 1 2 3.",
    )
    inference.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of diffusion sampling steps. If not set, uses the model default.",
    )

    runtime.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local model weights.",
    )
    runtime.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for AtlasFold and AtlasLM weights.",
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
        help=(
            "Maximum bucketed residue tokens per model call. Lower this if "
            "CUDA runs out of memory."
        ),
    )
    runtime.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit residue buckets. Defaults to the runner buckets: "
            "32, 64, 128, 192, 256, 384, 512, 640, 768, ..."
        ),
    )

    output.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    output.add_argument(
        "--save-confidence-arrays",
        action="store_true",
        help="Save raw pLDDT and PAE arrays for each sample as NPZ files.",
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


def load_sequences(input_fasta: Path) -> list[FoldingInput]:
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file does not exist: {input_fasta}")

    sequences: list[FoldingInput] = []
    for header, sequence in read_fasta(input_fasta):
        name = header.split()[0].strip()
        sequences.append(FoldingInput(name, sequence))
    if len(sequences) == 0:
        raise ValueError(f"No sequences found in FASTA file: {input_fasta}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in sequences:
        name = item.name
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "FASTA target names must be unique after normalization. "
            f"Duplicates: {sorted(set(duplicates))}"
        )

    return sorted(sequences, key=lambda item: (len(item.sequence), item.name))


def load_model(
    model_path: str | Path | None = None,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
) -> AtlasFold:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Local weight file does not exist: {model_path}")
        logger.info("Loading local weight path=%s, device=%s", model_path, device)
    else:
        logger.info(
            "Loading atlasfold-260703 from Hugging Face, cache_dir=%s, device=%s",
            cache_dir,
            device,
        )

    model = load_pretrained_model(
        "atlasfold-260703",
        device=device,
        cache_dir=cache_dir,
        model_path=model_path,
    )
    if not isinstance(model, AtlasFold):
        raise TypeError(f"Expected AtlasFold, got {type(model)!r}.")
    return model


def write_outputs(
    out_dir: Path,
    output: FoldingOutput,
    format: str,
    save_confidence_arrays: bool = False,
    save_distogram: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_records = []
    done_path = out_dir / "done.txt"
    if done_path.exists():
        done_path.unlink()

    if len(output.ranking) == 0:
        raise ValueError(f"No samples to rank for target {output.name!r}.")
    best_sample_idx = output.best_key

    for (seed, sample_idx), sample in sorted(output.outputs.items()):
        target_name = output.name
        sample_name = f"{target_name}_seed-{seed}_sample-{sample_idx}"

        sample_text = sample.to_mmcif() if format == "cif" else sample.to_pdb()
        confidence_scores = sample.confidence_scores

        with open(out_dir / f"{sample_name}_model.{format}", "w") as f:
            f.write(sample_text)
        with open(out_dir / f"{sample_name}_confidence.json", "w") as f:
            json.dump(confidence_scores, f, indent=2)
        if save_confidence_arrays:
            np.savez_compressed(
                out_dir / f"{sample_name}_confidence.npz",
                plddt=sample.plddt,
                pae=sample.pae,
            )

        record = {
            "seed": seed,
            "sample_index": sample_idx,
            "sample_name": sample_name,
            "avg_plddt": confidence_scores["avg_plddt"],
            "ptm": confidence_scores["ptm"],
        }
        sample_records.append(record)

        if (seed, sample_idx) == best_sample_idx:
            rank_name = f"{target_name}_ranked"
            best_record = record
            with open(out_dir / f"{rank_name}_model.{format}", "w") as f:
                f.write(sample_text)
            with open(out_dir / f"{rank_name}_confidence.json", "w") as f:
                json.dump(confidence_scores, f, indent=2)

    if save_distogram:
        if len(output.distogram_logits) == 0 or output.distogram_boundaries is None:
            raise ValueError(f"No distograms are available for target {output.name!r}.")
        for seed, logits in sorted(output.distogram_logits.items()):
            np.savez_compressed(
                out_dir / f"{output.name}_seed-{seed}_distogram.npz",
                logits=logits,
                boundaries=output.distogram_boundaries,
            )

    with open(out_dir / f"{output.name}_summary.csv", "w") as f:
        f.write("seed,sample_index,avg_plddt,ptm\n")
        for record in sample_records:
            f.write(
                f"{record['seed']},"
                f"{record['sample_index']},"
                f"{record['avg_plddt']:.3f},"
                f"{record['ptm']:.3f}\n"
            )
    done_path.touch()
    return best_record


def run(args: argparse.Namespace) -> None:
    setup_logging()

    sequences = load_sequences(args.input_fasta)
    logger.info(
        "Loaded %d sequences from %s. Length range: %d-%d.",
        len(sequences),
        args.input_fasta,
        len(sequences[0].sequence),
        len(sequences[-1].sequence),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter out completed targets unless --overwrite is set.
    if not args.overwrite:
        filtered = [
            item for item in sequences if not (out_dir / item.name / "done.txt").exists()
        ]
        num_skipped = len(sequences) - len(filtered)
        if num_skipped > 0:
            logger.info("Skipping %d completed targets.", num_skipped)
        sequences = filtered

    if len(sequences) == 0:
        logger.info("All targets are complete. Nothing to do.")
        return

    model = load_model(
        model_path=args.model_path,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    if args.no_kernel:
        model.set_forward_flags(use_cuequiv_kernels=False)

    # Set up the runner
    runner = FoldingRunner(model)
    num_steps = args.num_steps if args.num_steps is not None else "auto"
    if num_steps == "auto":
        logger.info(
            "Using automatic diffusion steps based on bucket length: "
            "20 (L <= 512), 30 (512 < L <= 1024), 100 (L > 1024)."
        )
        sampling_config = None
    else:
        logger.info("Using fixed diffusion steps: %d.", num_steps)
        sampling_config = SamplingConfig(num_steps=args.num_steps, chunk_size=5)

    logger.info(
        "Starting monomer inference: seeds=%s, num_samples=%d, "
        "num_recycles=%d, mlm_prob=%s, num_steps=%s, stochastic=%s, format=%s",
        args.seed,
        args.num_samples,
        args.num_recycles,
        args.mlm_prob,
        str(num_steps),
        args.stochastic,
        args.format,
    )

    num_finished = 0
    num_total = len(sequences)
    start = timer()
    batch_start = timer()
    try:
        for outputs in runner.fold_iter_batch(
            sequences,
            num_samples=args.num_samples,
            seeds=args.seed,
            num_recycles=args.num_recycles,
            mlm_prob=args.mlm_prob,
            stochastic=args.stochastic,
            sampling_config=sampling_config,
            length_buckets=args.length_buckets,
            max_tokens_per_batch=args.max_tokens_per_batch,
            return_distogram=args.save_distogram,
        ):
            batch_elapsed = timer() - batch_start
            time_per_target = batch_elapsed / len(outputs)
            batch_start = timer()

            for output in outputs:
                target_dir = out_dir / output.name
                best_record = write_outputs(
                    target_dir,
                    output,
                    args.format,
                    save_confidence_arrays=args.save_confidence_arrays,
                    save_distogram=args.save_distogram,
                )
                num_finished += 1
                logger.info(
                    "Completed %s (length=%d): "
                    "avg_plddt=%.3f, ptm=%.3f time=%.2f (%d/%d)",
                    output.name,
                    output.length,
                    best_record["avg_plddt"],
                    best_record["ptm"],
                    time_per_target,
                    num_finished,
                    num_total,
                )
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            logger.error(
                "CUDA out of memory during inference. Try lowering "
                "--max-tokens-per-batch."
            )
        raise

    elapsed = timer() - start
    logger.info(
        "Finished %d targets in %.1fs (%.1fs per target).",
        len(sequences),
        elapsed,
        elapsed / len(sequences),
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    args = create_parser().parse_args()
    run(args)
