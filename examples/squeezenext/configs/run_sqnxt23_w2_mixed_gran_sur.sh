

MAX_ACT=893.0
SUM_WEIGHT=321.0


for SIZE in 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)

  python3 train.py --workdir=../../sqnxt23_w2_mixed_${SIZE}_gran_sur_9 --config=squeezenext/configs/sqnxt23_w2_mixed_gran_sur.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/pretrained/sqnxt23_w2_${BITS}_pre_gran_sur_2/best
  if [ -d ../../sqnxt23_w2_mixed_${SIZE}_gran_sur_9/best ]; then
    python3 train.py --workdir=../../sqnxt23_w2_mixed_${SIZE}_finetune_gran_sur_9 --config=squeezenext/configs/sqnxt23_w2_mixed_finetune_gran_sur.py --config.pretrained_quant=../../sqnxt23_w2_mixed_${SIZE}_gran_sur_9/best
  else
    python3 train.py --workdir=../../sqnxt23_w2_mixed_${SIZE}_finetune_gran_sur_9 --config=squeezenext/configs/sqnxt23_w2_mixed_finetune_gran_sur.py --config.pretrained_quant=../../sqnxt23_w2_mixed_${SIZE}_gran_sur_9/
  fi

done