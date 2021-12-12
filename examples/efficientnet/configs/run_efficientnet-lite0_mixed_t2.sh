

MAX_ACT=150.528
SUM_WEIGHT=581.501


for SIZE in 5 4.75 4.5 4.25 4 3.75 3.5 3.25 3
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)

  rm -rf ../../efficientnet-lite0_mixed_${SIZE}
  mkdir ../../efficientnet-lite0_mixed_${SIZE}
  python3 train.py --workdir=../../efficientnet-lite0_mixed_${SIZE} --config=efficientnet/configs/efficientnet-lite0_mixed.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET}

done
