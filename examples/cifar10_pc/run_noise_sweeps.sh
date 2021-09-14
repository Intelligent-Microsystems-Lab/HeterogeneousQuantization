#!/bin/bash
module load python

# Several noise sweeps on Notre Dame CRC cluster.

MAIN_DIR="/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps"
USER="cschaef6"
GPU_Q="gpu@@joshi"
START="0."
STOP=".15"
STEP="0.01"

mkdir /afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps

# Predictive Coding Sweeps
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q --parameter weight_noise --network_type pc
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q  --parameter act_noise --network_type pc
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q --parameter err_inpt_noise --network_type pc
python3 sweep.py  --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q --parameter err_weight_noise --network_type pc

# Backpropagation Sweeps
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q --parameter weight_noise --network_type bp
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q  --parameter act_noise --network_type bp
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q --parameter err_inpt_noise --network_type bp
python3 sweep.py  --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --queue $GPU_Q --parameter err_weight_noise --network_type bp
