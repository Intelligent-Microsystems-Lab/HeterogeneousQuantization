
MAX_ACT=893.0
SUM_WEIGHT=321.0


for SIZE in 2 2.2 2.4 2.6 2.8 3 3.2 3.4 3.6 3.8 4
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)

  python3 train.py --workdir=../../sqnxt23_w2_mixed_${SIZE}_gran_7 --config=squeezenext/configs/sqnxt23_w2_mixed_gran.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/pretrained_hq/sqnxt_${BITS}_gran_pre_3/best
  if [ -d ../../sqnxt23_w2_mixed_${SIZE}_gran_7/best ]; then
    python3 train.py --workdir=../../sqnxt23_w2_mixed_${SIZE}_finetune_gran_7 --config=squeezenext/configs/sqnxt23_w2_mixed_finetune_gran.py --config.pretrained_quant=../../sqnxt23_w2_mixed_${SIZE}_gran_7/best
  else
    python3 train.py --workdir=../../sqnxt23_w2_mixed_${SIZE}_finetune_gran_7 --config=squeezenext/configs/sqnxt23_w2_mixed_finetune_gran.py --config.pretrained_quant=../../sqnxt23_w2_mixed_${SIZE}_gran_7/
  fi

done
