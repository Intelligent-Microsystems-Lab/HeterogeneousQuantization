
for SCALE in 0.01 0.001 0.0001 0.00001
do

  rm -rf ../../efficientnet-lite0_noise_bwd_${SCALE}_${SEED}
  mkdir ../../efficientnet-lite0_noise_bwd_${SCALE}_${SEED}
  python3 train.py --workdir=../../efficientnet-lite0_noise_bwd_${SCALE}_${SEED} --config=efficientnet/configs/efficientnet-lite0_w3a3_round_bwd_noise.py  --config.seed=14339038 --config.quant.g_scale=${SCALE}
done
