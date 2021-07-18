#!/bin/bash
module load python cuda/11.0
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/11.0"
cd smile_pc.py
python train.py
cd ../smile_rtrl.py
python train.py
cd ../smile_bptt.py
python train.py
cd ../smile_pc_rtrl.py
python train.py
cd ..