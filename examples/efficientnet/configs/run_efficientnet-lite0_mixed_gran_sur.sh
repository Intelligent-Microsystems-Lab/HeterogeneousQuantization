

MAX_ACT=835.0
SUM_WEIGHT=577.0


for SIZE in 3.5 3.6 3.7 3.8 3.9 4 4.1 4.2
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)

  python3 train.py --workdir=../../efficientnet-lite0_mixed_${SIZE}_gran_sur_12 --config=efficientnet/configs/efficientnet-lite0_mixed_gran_sur.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/pretrained_hq/enet_${BITS}_gran_sur_pre_3/best
  if [ -d ../../efficientnet-lite0_mixed_${SIZE}_gran_sur_9/best ]; then
    python3 train.py --workdir=../../efficientnet-lite0_mixed_${SIZE}_gran_sur_finetune_12 --config=efficientnet/configs/efficientnet-lite0_mixed_gran_sur_finetune.py --config.pretrained_quant=../../efficientnet-lite0_mixed_${SIZE}_gran_sur_12/best
  else
    python3 train.py --workdir=../../efficientnet-lite0_mixed_${SIZE}_gran_sur_finetune_12 --config=efficientnet/configs/efficientnet-lite0_mixed_gran_sur_finetune.py --config.pretrained_quant=../../efficientnet-lite0_mixed_${SIZE}_gran_sur_12/
  fi

done
