import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import torch

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold, SamplingConfig
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
    parser.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help="Path to the input FASTA file.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to local AtlasFold weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Disable cuequivariance kernels.",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=4,
        help="Number of recycling iterations.",
    )
    parser.add_argument(
        "--mlm-prob",
        type=float,
        default=0.15,
        help="LM masking probability used during recycling.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic LM features during all recycling iterations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate per input sequence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[1],
        help="Random seed(s) for inference. Example: --seed 1 2 3.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of diffusion sampling steps. If not set, uses the model default.",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=1024,
        help=(
            "Maximum bucketed residue tokens per model call. Lower this if "
            "CUDA runs out of memory."
        ),
    )
    parser.add_argument(
        "--length-buckets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional explicit residue buckets. Defaults to the runner buckets: "
            "32, 64, 128, 192, 256, 384, 512, 640, 768, ..."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format for sample and ranked outputs.",
    )
    parser.add_argument(
        "--use-fasta-chain-ids",
        action="store_true",
        help=(
            "Read optional chain_id=... metadata from FASTA headers. When set, "
            "headers may only contain the target name and chain_id=..."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def normalize_target_name(header: str) -> str:
    fields = header.split()
    if len(fields) == 0:
        raise ValueError(f"Invalid FASTA header: {header!r}")
    name = fields[0]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    if not safe_name:
        raise ValueError(f"Invalid FASTA header: {header!r}")
    return safe_name


def parse_fasta_header(
    header: str,
    use_fasta_chain_ids: bool,
) -> tuple[str, str | None]:
    fields = header.split()
    if len(fields) == 0:
        raise ValueError(f"Invalid FASTA header: {header!r}")

    name = normalize_target_name(header)
    chain_id = None
    if use_fasta_chain_ids:
        for field in fields[1:]:
            key, sep, value = field.partition("=")
            if not sep or key != "chain_id":
                raise ValueError(
                    f"Invalid FASTA header for {name}: {header!r}. When "
                    "--use-fasta-chain-ids is set, monomer headers may only "
                    "contain the target name and chain_id=..."
                )
            if chain_id is not None:
                raise ValueError(f"FASTA header for {name} has multiple chain ID fields.")
            chain_id = value

    return name, chain_id


def load_sequences(
    input_fasta: Path,
    format: str = "cif",
    use_fasta_chain_ids: bool = False,
) -> list[FoldingInput]:
    if not input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file does not exist: {input_fasta}")

    sequences: list[FoldingInput] = []
    for header, sequence in read_fasta(input_fasta):
        name, chain_id = parse_fasta_header(header, use_fasta_chain_ids)
        sequences.append(FoldingInput(name, sequence, chain_id or "A"))
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
    model_path: str | Path,
    device: str | torch.device | None = None,
    disable_kernel: bool = False,
) -> AtlasFold:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Local weight file does not exist: {model_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    logger.info("Loading weight path=%s, device=%s", model_path, device)
    state_dict = torch.load(model_path, map_location="cpu")

    model = AtlasFold.from_pretrained(state_dict=state_dict, device=device)
    model.set_forward_flags(use_cuequiv_kernels=not disable_kernel)
    return model


def write_outputs(
    out_dir: Path,
    output: FoldingOutput,
    format: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_records = []
    done_path = out_dir / "done.txt"
    if done_path.exists():
        done_path.unlink()

    if len(output.ranking) == 0:
        raise ValueError(f"No samples to rank for target {output.name!r}.")
    best_sample_idx: tuple[int, int] = output.ranking[0]

    for (seed, sample_idx), sample in sorted(output.outputs.items()):
        target_name = output.name
        sample_name = f"{target_name}_seed-{seed}_sample-{sample_idx}"

        sample_text = sample.to_mmcif() if format == "cif" else sample.to_pdb()
        confidence_scores = sample.confidence_scores

        with open(out_dir / f"{sample_name}_model.{format}", "w") as f:
            f.write(sample_text)
        with open(out_dir / f"{sample_name}_confidence.json", "w") as f:
            json.dump(confidence_scores, f, indent=2)

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

    sequences = load_sequences(args.input_fasta, args.format, args.use_fasta_chain_ids)
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

    model = load_model(args.model_path, args.device, args.no_kernel)
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
        for outputs in runner.iter_fold_batch(
            sequences,
            num_samples=args.num_samples,
            seeds=args.seed,
            num_recycles=args.num_recycles,
            mlm_prob=args.mlm_prob,
            stochastic=args.stochastic,
            sampling_config=sampling_config,
            length_buckets=args.length_buckets,
            max_tokens_per_batch=args.max_tokens_per_batch,
        ):
            batch_elapsed = timer() - batch_start
            time_per_target = batch_elapsed / len(outputs)
            batch_start = timer()

            for output in outputs:
                target_dir = out_dir / output.name
                best_record = write_outputs(target_dir, output, args.format)
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


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
