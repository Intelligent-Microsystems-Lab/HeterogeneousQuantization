

for BITS in 3 #3 4 5 6 7 8
do
  for SEED in 193012823 235899598 8627169 103372330 14339038
  do
    rm -rf ../../efficientnet-lite0_int${BITS}_seed${SEED}
    mkdir ../../efficientnet-lite0_int${BITS}_seed${SEED}
    python3 train.py --workdir=../../efficientnet-lite0_int${BITS}_seed${SEED} --config=efficientnet/configs/efficientnet-lite0_w3a3.py  --config.quant.bits=${BITS} --config.seed=${SEED}
  done
done
