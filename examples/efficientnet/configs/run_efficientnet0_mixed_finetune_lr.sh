MAX_ACT=920.0
SUM_WEIGHT=595.0
SIZE=3.0

for LR in 0.00125 0.000125 0.0000125 0.00000125
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)

  rm -rf ../../efficientnet-lite0_mixed_finetunelr_${LR}
  mkdir ../../efficientnet-lite0_mixed_finetunelr_${LR}
  python3 train.py --workdir=../../efficientnet-lite0_mixed_finetunelr_${LR} --config=efficientnet/configs/efficientnet-lite0_mixed.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.finetune.learning_rate=${LR}

done