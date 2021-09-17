#!/bin/bash
module load python

# Several noise sweeps on Notre Dame CRC cluster.
MAIN_DIR="/afs/crc.nd.edu/user/c/cschaef6/mnist_noise_sweeps"
USER="cschaef6"
GPU_Q="gpu@@joshi"
START="0."
STOP=".15"
STEP="0.01"


LIST="0.125000,0.055556,0.031250,0.020000,0.013889,0.010204,0.007812,0.006173,0.005000,0.004132,0.003472,0.002959,0.002551,0.002222,0.001953,0.000000"

mkdir $MAIN_DIR

# Backpropagation Sweeps
python3 sweep.py --result_dir $MAIN_DIR/weight_noise_bp --list $LIST --user $USER --trials 5 --queue $GPU_Q --parameter weight_noise --network_type bp
python3 sweep.py --result_dir $MAIN_DIR/act_noise_bp --list $LIST --user $USER --trials 5 --queue $GPU_Q  --parameter act_noise --network_type bp
python3 sweep.py --result_dir $MAIN_DIR/err_inpt_noise_bp --list $LIST --user $USER --trials 5 --queue $GPU_Q --parameter err_inpt_noise --network_type bp
python3 sweep.py  --result_dir $MAIN_DIR/err_weight_noise_bp --list $LIST --user $USER --trials 5 --queue $GPU_Q --parameter err_weight_noise --network_type bp

# # Predictive Coding Sweeps
# python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --list $LIST --user $USER --trials 5 --queue $GPU_Q --parameter weight_noise --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/act_noise_pc --list $LIST --user $USER --trials 5 --queue $GPU_Q  --parameter act_noise --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/err_inpt_noise_pc --list $LIST --user $USER --trials 5 --queue $GPU_Q --parameter err_inpt_noise --network_type pc
# python3 sweep.py  --result_dir $MAIN_DIR/err_weight_noise_pc --list $LIST --user $USER --trials 5 --queue $GPU_Q --parameter err_weight_noise --network_type pc