import time
from pathlib import Path

import torch
from tqdm import tqdm

from atlasfold.data.fasta import read_fasta
from atlasfold.model import AtlasFold, AtlasFoldConfig, SamplingConfig
from atlasfold.runner import FoldingRunner

CKPT = "/mnt/parallel_storage/wykim_lab/icl_shwan/train_results/stage1/stage1/AtlasFold/fuj7x3dw/checkpoints/epoch0031_step00032000_lddt0.8199.ckpt"
CKPT_NAME = "32k"
PRESET = "full"
NUM_RECYCLES = 3
NUM_STEPS = 200
SIGMA_MAX = 128

ROOT_DIR = Path("./outputs/cameo2022/")
EXP_NAME = f"{CKPT_NAME}_{PRESET}_steps{NUM_STEPS}_sigma{SIGMA_MAX}"


@torch.inference_mode()
def main():
    # Load model
    device = torch.device("cuda")
    config = AtlasFoldConfig()
    state_dict = torch.load(CKPT, map_location="cuda")
    model_state_dict = {
        k.removeprefix("model."): v for k, v in state_dict["state_dict"].items()
    }
    ema_state_dict = state_dict["ema"]["params"]
    model_state_dict.update(ema_state_dict)
    del state_dict
    model = AtlasFold.from_pretrained(
        state_dict=model_state_dict,
        config=config,
        device=device,
        dtype=torch.bfloat16,
    )

    # Create output directory
    OUT_DIR = ROOT_DIR / EXP_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load sequences
    sequences = read_fasta("./assets/test/cameo2022.fasta")
    sequences.sort(key=lambda x: (len(x[1]), x[0]))

    # Run folding
    runner = FoldingRunner(model)
    sampling_config = SamplingConfig(num_steps=NUM_STEPS, sigma_max=SIGMA_MAX)

    for header, seq in tqdm(sequences):
        print()
        name = header.split()[0]
        out_path = OUT_DIR / f"{name}_seed-1_sample-0.cif"
        if out_path.exists():
            print(f"Output for {name} already exists, skipping...")
            continue

        print(f"Folding {name} with length {len(seq)}...")

        start_time = time.time()
        out = runner.fold(
            name=name,
            sequence=seq,
            num_samples=1,
            preset=PRESET,
            seed=1,
            num_recycles=NUM_RECYCLES,
            sampling_config=sampling_config,
        )[0]
        elapsed = time.time() - start_time
        print(f"Length {len(seq)}: {elapsed:.2f} seconds")

        with open(out_path, "w") as f:
            f.write(out.to_mmcif())


if __name__ == "__main__":
    main()
