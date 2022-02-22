

MAX_ACT=255.0
SUM_WEIGHT=1460.0


for SIZE in 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4
do

  ACT_TARGET=$(echo "$MAX_ACT*$SIZE" | bc -l)
  WEIGHT_TARGET=$(echo "$SUM_WEIGHT*$SIZE" | bc -l)
  BITS=$(echo "scale=0; ($SIZE)/1 + 2" | bc -l)

  python3 train.py --workdir=../../resnet18_mixed_${SIZE}_9 --config=resnet/configs/resnet18_mixed.py  --config.quant_target.weight_mb=${WEIGHT_TARGET} --config.quant_target.act_mb=${ACT_TARGET} --config.quant.w_bits=${BITS} --config.quant.a_bits=${BITS} --config.pretrained_quant=gs://imagenet_clemens/pretrained/resnet18_${BITS}_pre_2/best
  if [ -d ../../resnet18_mixed_${SIZE}_9/best ]; then
    python3 train.py --workdir=../../resnet18_mixed_${SIZE}_finetune_9 --config=resnet/configs/resnet18_mixed_finetune.py --config.pretrained_quant=../../resnet18_mixed_${SIZE}_9/best
  else
    python3 train.py --workdir=../../resnet18_mixed_${SIZE}_finetune_9 --config=resnet/configs/resnet18_mixed_finetune.py --config.pretrained_quant=../../resnet18_mixed_${SIZE}_9/
  fi

done
