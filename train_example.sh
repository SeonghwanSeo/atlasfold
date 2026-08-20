python scripts/train_monomer_ipa.py   \
  configs/monomer_ipa/train_stage1.yaml \
  --init_weight weights/atlasfold-stage1.pth \
  --wandb \
  --compile \
  --override \
    train.data.batch_size=2
