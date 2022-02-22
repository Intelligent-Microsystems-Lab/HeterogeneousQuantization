

MAX_ACT=993.0
SUM_WEIGHT=434.0


for SIZE in 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)

  python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_sur_9 --config=mobilenetv2/configs/mobilenetv2_mixed_sur.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/pretrained/mbnet_${BITS}_pre_sur_2/best
  python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_finetune_9 --config=mobilenetv2/configs/mobilenetv2_mixed_finetune_sur.py --config.pretrained_quant=../../mbnetv2_mixed_${SIZE}_sur_9/best

done
