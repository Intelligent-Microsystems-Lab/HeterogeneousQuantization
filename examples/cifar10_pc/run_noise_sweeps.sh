#!/bin/bash
module load python

# Several noise sweeps on Notre Dame CRC cluster.

# Predictive Coding Sweeps
python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/weight_noise_pc --start 0 --stop .3 --step .01 --parameter weight_noise --network_type pc --user cschaef6 --queue gpu@@joshi
python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/act_noise_pc --start 0 --stop .3 --step .01 --parameter act_noise --network_type pc --user cschaef6 --queue gpu@@joshi
python3 python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_inpt_noise_pc --start 0 --stop .3 --step .01 --parameter err_inpt_noise --network_type pc --user cschaef6 --queue gpu@@joshi
python3 python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_weight_noise_pc --start 0 --stop .3 --step .01 --parameter err_weight_noise --network_type pc --user cschaef6 --queue gpu@@joshi

# Backpropagation Sweeps
python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/weight_noise_bp --start 0 --stop .3 --step .01 --parameter weight_noise --network_type bp --user cschaef6 --queue gpu@@joshi
python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/act_noise_bp --start 0 --stop .3 --step .01 --parameter act_noise --network_type bp --user cschaef6 --queue gpu@@joshi
python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_inpt_noise_bp --start 0 --stop .3 --step .01 --parameter err_inpt_noise --network_type bp --user cschaef6 --queue gpu@@joshi
python3 sweep.py --result_dir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps/err_weight_noise_bp --start 0 --stop .3 --step .01 --parameter err_weight_noise --network_type bp --user cschaef6 --queue gpu@@joshi
