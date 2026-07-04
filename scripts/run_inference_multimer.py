import argparse
import json
import logging
import sys
from pathlib import Path
from timeit import default_timer as timer

import torch

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold_Multimer, SamplingConfig
from atlasfold.runner_multimer import (
    MultimerFoldingOutput,
    MultimerFoldingRunner,
    MultimerInput,
)

logger = logging.getLogger("atlasfold.multimer")


def setup_logging() -> None:
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AtlasFold-Multimer inference on a FASTA file. Each FASTA "
            "record is treated as one complex, with ':' separating chains."
        )
    )
    parser.add_argument(
        "-i",
        "--input-fasta",
        type=Path,
        required=True,
        help=(
            "Input FASTA file. Each FASTA record is treated as one complex, "
            "with ':' separating chains."
        ),
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where predictions will be written.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to an AtlasFold-Multimer checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
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
        default=10,
        help="Number of recycling iterations.",
    )
    parser.add_argument(
        "--mlm-prob",
        type=float,
        default=0.15,
        help="LM masking probability used during recycling.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of diffusion samples to generate.",
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
        default=200,
        help="Number of diffusion sampling steps.",
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
            "Read optional chain_ids=... metadata from FASTA headers. "
            "The IDs must match ':'-separated chains in order."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute targets even when outputs already exist.",
    )
    return parser


def safe_target_name(name: str) -> str:
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    if not safe_name:
        raise ValueError(f"Invalid target name: {name!r}")
    return safe_name


def parse_fasta_header(
    header: str,
    use_fasta_chain_ids: bool,
) -> tuple[str, list[str] | None]:
    fields = header.split()
    if len(fields) == 0:
        raise ValueError(f"Invalid FASTA header: {header!r}")

    name = safe_target_name(fields[0])
    chain_ids = None
    if use_fasta_chain_ids:
        for field in fields[1:]:
            key, sep, value = field.partition("=")
            if not sep or key != "chain_ids":
                raise ValueError(
                    f"Invalid FASTA header for {name}: {header!r}. When "
                    "--use-fasta-chain-ids is set, multimer headers may only "
                    "contain the target name and chain_ids=..."
                )
            if chain_ids is not None:
                raise ValueError(f"FASTA header for {name} has multiple chain ID fields.")
            chain_ids = value.split(",")

    return name, chain_ids


def validate_chain_ids(
    target_name: str,
    sequences: list[str],
    chain_ids: list[str] | None,
) -> None:
    if chain_ids is None:
        return
    if len(chain_ids) != len(sequences):
        raise ValueError(
            f"FASTA target {target_name} has {len(sequences)} chains but "
            f"{len(chain_ids)} chain IDs."
        )


def split_chain_sequences(sequence: str) -> list[str]:
    sequences = [part for part in sequence.split(":") if part.strip()]
    if len(sequences) == 0:
        raise ValueError("FASTA target has no non-empty chains.")
    return sequences


def load_inputs(args: argparse.Namespace) -> list[MultimerInput]:
    if not args.input_fasta.exists():
        raise FileNotFoundError(f"Input FASTA file does not exist: {args.input_fasta}")

    inputs: list[MultimerInput] = []
    for header, sequence in read_fasta(args.input_fasta):
        name, chain_ids = parse_fasta_header(header, args.use_fasta_chain_ids)
        sequences = split_chain_sequences(sequence)
        validate_chain_ids(name, sequences, chain_ids)
        inputs.append(MultimerInput(name, sequences, chain_ids))
    if len(inputs) == 0:
        raise ValueError(f"No sequences found in FASTA file: {args.input_fasta}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for item in inputs:
        name = item.name
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            "FASTA target names must be unique after normalization. "
            f"Duplicates: {sorted(set(duplicates))}"
        )

    return inputs


def load_model(args: argparse.Namespace):
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {args.checkpoint}")

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    logger.info("Loading checkpoint: path=%s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {
            k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()
        }
    else:
        state_dict = {k.removeprefix("model."): v for k, v in checkpoint.items()}

    model = AtlasFold_Multimer.from_pretrained(
        state_dict=state_dict,
        device=device,
    )
    model.set_forward_flags(use_cuequiv_kernels=not args.no_kernel)
    return model


def write_outputs(
    out_dir: Path,
    output: MultimerFoldingOutput,
    format: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_records = []

    # Remove the "done.txt" file if it exists, since we are overwriting the outputs.
    done_path = out_dir / "done.txt"
    if done_path.exists():
        done_path.unlink()

    best_sample_idx: tuple[int, int] = output.ranking[0]
    for (seed, sample_idx), sample in sorted(output.outputs.items()):
        target_name = output.name
        sample_name = f"{target_name}_seed-{seed}_sample-{sample_idx}"

        sample_text = sample.to_mmcif() if format == "cif" else sample.to_pdb()
        confidence_scores = sample.confidence_scores

        with open(out_dir / f"{sample_name}.{format}", "w") as f:
            f.write(sample_text)
        with open(out_dir / f"{sample_name}_confidence.json", "w") as f:
            json.dump(confidence_scores, f, indent=2)

        record = {
            "seed": seed,
            "sample_index": sample_idx,
            "sample_name": sample_name,
            "score": sample.ranking_score,
            "ranking_score": confidence_scores["complex"]["ranking_score"],
            "ptm": confidence_scores["complex"]["ptm"],
            "iptm": confidence_scores["complex"]["iptm"],
            "avg_plddt": confidence_scores["complex"]["avg_plddt"],
            "avg_pde": confidence_scores["complex"]["avg_pde"],
        }

        sample_records.append(record)

        if (seed, sample_idx) == best_sample_idx:
            rank_name = f"{target_name}_ranked"
            best_record = record
            with open(out_dir / f"{rank_name}.{format}", "w") as f:
                f.write(sample_text)
            with open(out_dir / f"{rank_name}_confidence.json", "w") as f:
                json.dump(confidence_scores, f, indent=2)

    with open(out_dir / f"{output.name}_summary.csv", "w") as f:
        f.write("seed,sample_index,ranking_score,ptm,iptm,avg_pde,avg_plddt\n")
        for record in sample_records:
            f.write(
                f"{record['seed']},"
                f"{record['sample_index']},"
                f"{record['ranking_score']:.3f},"
                f"{record['ptm']:.3f},"
                f"{record['iptm']:.3f},"
                f"{record['avg_pde']:.3f},"
                f"{record['avg_plddt']:.3f}\n"
            )

    # Create a "done.txt" file to indicate that the target is complete.
    done_path.touch()

    return best_record


def run(args: argparse.Namespace) -> None:
    setup_logging()
    inputs = load_inputs(args)
    num_residues = [item.length for item in inputs]
    logger.info(
        "Loaded %d multimer target(s). Residue range: %d-%d.",
        len(inputs),
        min(num_residues),
        max(num_residues),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter out completed targets unless --overwrite is set.
    if not args.overwrite:
        filtered = [
            item for item in inputs if not (out_dir / item.name / "done.txt").exists()
        ]
        num_skipped = len(inputs) - len(filtered)
        if num_skipped > 0:
            logger.info("Skipping %d completed targets.", num_skipped)
        inputs = filtered

    if len(inputs) == 0:
        logger.info("All targets are complete. Nothing to do.")
        return

    model = load_model(args)
    runner = MultimerFoldingRunner(model)
    sampling_config = SamplingConfig(num_steps=args.num_steps, chunk_size=5)

    logger.info(
        "Starting multimer inference: seeds=%s, num_samples=%d, "
        "num_recycles=%d, mlm_prob=%s, num_steps=%d, format=%s",
        args.seed,
        args.num_samples,
        args.num_recycles,
        args.mlm_prob,
        args.num_steps,
        args.format,
    )

    num_finished = 0
    num_total = len(inputs)
    start = timer()
    batch_start = timer()
    for outputs in runner.iter_fold_batch(
        inputs,
        num_samples=args.num_samples,
        seeds=args.seed,
        num_recycles=args.num_recycles,
        mlm_prob=args.mlm_prob,
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
                "ptm=%.3f, iptm=%.3f, avg_plddt=%.3f time=%.2f (%d/%d)",
                output.name,
                output.length,
                best_record["ptm"],
                best_record["iptm"],
                best_record["avg_plddt"],
                time_per_target,
                num_finished,
                num_total,
            )
    elapsed = timer() - start
    logger.info(
        "Finished %d target(s) in %.1fs.",
        len(inputs),
        elapsed,
    )


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    main()
