

MAX_ACT=255.0
SUM_WEIGHT=1460.0


for SIZE in 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)


  result=$(gsutil ls gs://imagenet_clemens/efficient_frontier/resnet18_mixed_${SIZE}_sur_9/best | wc -l)

  if [[ $result == 0 ]]; then
    python3 train.py --workdir=../../resnet18_mixed_${SIZE}_gran_sur_8 --config=resnet/configs/resnet18_mixed_gran_sur.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.w_bits=${BITS} --config.quant.a_bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/efficient_frontier/resnet18_mixed_${SIZE}_sur_9
  else
    python3 train.py --workdir=../../resnet18_mixed_${SIZE}_gran_sur_8 --config=resnet/configs/resnet18_mixed_gran_sur.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.w_bits=${BITS} --config.quant.a_bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/efficient_frontier/resnet18_mixed_${SIZE}_sur_9/best
  fi



  
  if [ -d ../../resnet18_mixed_${SIZE}_gran_sur_8/best ]; then
    python3 train.py --workdir=../../resnet18_mixed_${SIZE}_finetune_gran_sur_8 --config=resnet/configs/resnet18_mixed_finetune_gran_sur.py --config.pretrained_quant=../../resnet18_mixed_${SIZE}_gran_sur_8/best
  else
    python3 train.py --workdir=../../resnet18_mixed_${SIZE}_finetune_gran_sur_8 --config=resnet/configs/resnet18_mixed_finetune_gran_sur.py --config.pretrained_quant=../../resnet18_mixed_${SIZE}_gran_sur_8/
  fi

done
