

MAX_ACT=920.0
SUM_WEIGHT=595.0

# 1 1.25 1.5 1.75
for SIZE in 2 2.25 2.5 2.75 3.0
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)

  rm -rf ../../efficientnet-lite0_mixed_${SIZE}
  mkdir ../../efficientnet-lite0_mixed_${SIZE}
  python3 train.py --workdir=../../efficientnet-lite0_mixed_${SIZE} --config=efficientnet/configs/efficientnet-lite0_mixed.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET}

done
