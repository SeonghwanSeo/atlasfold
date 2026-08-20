python ./scripts/train_monomer_ipa.py \
  configs/monomer_ipa/train_stage1.yaml \
  --init_weight weights/atlasfold-stage1.pth \
  --experiment_name 'v2-stage1-256' \
  --out_dir '/mnt/parallel_storage/wykim_lab/icl_shwan/entire-train-log/monomer-ipa/' \
  --wandb \
  --compile \
  --override train.data.batch_size=4
