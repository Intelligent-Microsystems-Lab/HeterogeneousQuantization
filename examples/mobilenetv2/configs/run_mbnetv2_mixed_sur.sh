

MAX_ACT=993.0
SUM_WEIGHT=434.0


for SIZE in 2 2.2 2.4 2.6 2.8 3 3.2 3.4 3.6 3.8 4
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)

  python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_sur_7 --config=mobilenetv2/configs/mobilenetv2_mixed_sur.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/pretrained_hq/mbnet_${BITS}_sur_pre_3/best
  if [ -d ../../mbnetv2_mixed_${SIZE}_sur_7/best ]; then
    python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_finetune_7 --config=mobilenetv2/configs/mobilenetv2_mixed_finetune_sur.py --config.pretrained_quant=../../mbnetv2_mixed_${SIZE}_sur_7/best
  else
    python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_finetune_7 --config=mobilenetv2/configs/mobilenetv2_mixed_finetune_sur.py --config.pretrained_quant=../../mbnetv2_mixed_${SIZE}_sur_7/
  fi

done
