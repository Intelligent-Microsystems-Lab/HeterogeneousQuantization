#!/bin/bash
module load python cuda/10.2
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"
cd copytask_bptt
python train.py
cd ../copytask_rtrl
python train.py
cd ../copytask_pc
python train.py
cd ..