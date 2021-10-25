
for BITS in 2 3 4 5 6 7 8
do
  rm -rf ../../../efficientnet-lite0_int${BITS}_dynamic_gaussian
  mkdir ../../../efficientnet-lite0_int${BITS}_dynamic_gaussian
  python3 train.py --workdir=../../../efficientnet-lite0_int${BITS}_dynamic_gaussian --config=configs/efficientnet-lite0_w8a8_dynamic_gaussian.py  --config.quant.bits=${BITS}
done
