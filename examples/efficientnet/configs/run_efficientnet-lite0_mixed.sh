

MAX_ACT=0.150528
SUM_WEIGHT=0.581501

for SIZE in 0.5 0.75 1 1.25 1.5 1.75 2 2.25 2.5 2.75
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)

  rm -rf ../../../efficientnet-lite0_mixed_${SIZE}
  mkdir ../../../efficientnet-lite0_mixed_${SIZE}
  python3 train.py --workdir=../../../efficientnet-lite0_mixed_${SIZE} --config=configs/efficientnet-lite0_mixed.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET}

done