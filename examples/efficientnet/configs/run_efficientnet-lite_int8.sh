
for NET in 0 1 2 3 4
do
  rm -rf ../../../efficientnet-lite${NET}_int8
  mkdir ../../../efficientnet-lite${NET}_int8
  python3 train.py --workdir=../../../efficientnet-lite${NET}_int8 --config=configs/efficientnet-lite${NET}_w8a8_dynamic.py --config.num_epochs=1
done
