

MAX_ACT=993.0
SUM_WEIGHT=434.0


for SIZE in 4 3.9 3.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1 3 2.9 2.8 2.7 2.6 2.5 2.4 2.3 2.2 2.1 2
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)


  result=$(gsutil ls gs://imagenet_clemens/efficient_frontier/mbnetv2_mixed_${SIZE}_9/best | wc -l)

  if [[ $result == 0 ]]; then
    python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_gran_8 --config=mobilenetv2/configs/mobilenetv2_mixed_gran.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/efficient_frontier/mbnetv2_mixed_${SIZE}_9
  else
    python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_gran_8 --config=mobilenetv2/configs/mobilenetv2_mixed_gran.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/efficient_frontier/mbnetv2_mixed_${SIZE}_9/best
  fi


  
  if [ -d ../../mbnetv2_mixed_${SIZE}_gran_9/best ]; then
    python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_finetune_gran_8 --config=mobilenetv2/configs/mobilenetv2_mixed_gran_finetune.py --config.pretrained_quant=../../mbnetv2_mixed_${SIZE}_gran_8/best
  else
    python3 train.py --workdir=../../mbnetv2_mixed_${SIZE}_finetune_gran_8 --config=mobilenetv2/configs/mobilenetv2_mixed_gran_finetune.py --config.pretrained_quant=../../mbnetv2_mixed_${SIZE}_gran_8/
  fi

done
