
for CONFIG in 1_15 2_15 3_15
do

  rm -rf ../../efficientnet-lite0_randaug_${CONFIG}
  mkdir ../../efficientnet-lite0_randaug_${CONFIG}
  python3 train.py --workdir=../../efficientnet-lite0_randaug_${CONFIG} --config=efficientnet/configs/efficientnet-lite0.py  --config.augment_name=randaugment_{$CONFIG}

done
