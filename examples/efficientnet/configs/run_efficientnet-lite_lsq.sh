
rm -rf ../../../efficientnet-lite0_lsq_int2
mkdir ../../../efficientnet-lite0_lsq_int2
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int2 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=2 --config.batch_norm_epsilon=1e-2

rm -rf ../../../efficientnet-lite0_lsq_int3
mkdir ../../../efficientnet-lite0_lsq_int3
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int3 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=3

rm -rf ../../../efficientnet-lite0_lsq_int4
mkdir ../../../efficientnet-lite0_lsq_int4
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int4 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=4

rm -rf ../../../efficientnet-lite0_lsq_int5
mkdir ../../../efficientnet-lite0_lsq_int5
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int5 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=5

rm -rf ../../../efficientnet-lite0_lsq_int6
mkdir ../../../efficientnet-lite0_lsq_int6
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int6 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=6

rm -rf ../../../efficientnet-lite0_lsq_int7
mkdir ../../../efficientnet-lite0_lsq_int7
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int7 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=7 --config.learning_rate=0.001

rm -rf ../../../efficientnet-lite0_lsq_int8
mkdir ../../../efficientnet-lite0_lsq_int8
python3 train.py --workdir=../../../efficientnet-lite0_lsq_int8 --config=configs/efficientnet-lite0_w4a4_lsq.py  --config.quant.bits=8 --config.learning_rate=0.001