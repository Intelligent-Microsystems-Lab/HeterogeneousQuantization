
for BITS in 2 3 4 5 6 7 8
do
  rm -rf ../../../efficientnet-lite0_int${BITS}_static_double_mean
  mkdir ../../../efficientnet-lite0_int${BITS}_static_double_mean
  python3 train.py --workdir=../../../efficientnet-lite0_int${BITS}_static_double_mean --config=configs/efficientnet-lite0_w8a8_static_double_mean.py  --config.quant.bits=${BITS}
done
