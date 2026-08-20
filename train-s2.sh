python ./scripts/train_monomer_ipa.py \
  configs/monomer_ipa/train_stage2.yaml \
  --experiment_name 'v2-stage2-384-run1' \
  --resume_from_checkpoint '/mnt/parallel_storage/wykim_lab/icl_shwan/entire-train-log/monomer-ipa/v2-stage1-256/AtlasFold_IPA/4i3zfg87/checkpoints/epoch0076_step00038500.ckpt' \
  --out_dir '/mnt/parallel_storage/wykim_lab/icl_shwan/entire-train-log/monomer-ipa/' \
  --wandb \
  --compile \
  --override \
    train.data.batch_size=4 \
    train.load_opt_state=false \
    train.trainer.limit_train_batches=500
