#!/bin/bash
module load python cuda/10.2
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"
cd shakespeare_bptt
python train.py
cd ../shakespeare_rtrl
python train.py
cd ../shakespeare_pc
python train.py
cd ..