
for BITS in 2 #3 4 5 6 7 8
do
  for LR in 0.0001 0.001 0.01
  do
    rm -rf ../../efficientnet-lite0_int${BITS}_lr${LR}
    mkdir ../../efficientnet-lite0_int${BITS}_lr${LR}
    python3 train.py --workdir=../../efficientnet-lite0_int${BITS}_lr${LR} --config=configs/efficientnet/efficientnet-lite0_w2a2.py  --config.quant.bits=${BITS} --config.learning_rate=${LR}
  done
done