
for NET in 0 1 2 3 4
do
  rm -rf ../../../efficientnet-lite${NET}_fp32
  mkdir ../../../efficientnet-lite${NET}_fp32
  python3 train.py --workdir=../../../efficientnet-lite${NET}_fp32 --config=configs/efficientnet-lite${NET}.py
done
