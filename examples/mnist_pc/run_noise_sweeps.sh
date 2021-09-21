#!/bin/bash
module load python

# Several noise sweeps on Notre Dame CRC cluster.
MAIN_DIR="/afs/crc.nd.edu/user/c/cschaef6/cifar10_noise_sweeps_t6"
USER="cschaef6"
GPU_Q="gpu@@joshi"
START="0."
STOP=".35"
STEP="0.025"
TRIALS="3"


#LIST="0.125000,0.055556,0.031250,0.020000,0.013889,0.010204,0.007812,0.006173,0.005000,0.004132,0.003472,0.002959,0.002551,0.002222,0.001953,0.000000"

#LIST="12,10,8,7,6,5,4,3,2"

LIST="12,11,10,9,8,7,6,5"

mkdir $MAIN_DIR

# # Predictive Coding Noise Sweeps
# python3 sweep.py --result_dir $MAIN_DIR/weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_noise --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/act_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_noise --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/err_inpt_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_inpt_noise --network_type pc
# python3 sweep.py  --result_dir $MAIN_DIR/err_weight_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_weight_noise --network_type pc

# python3 sweep.py --result_dir $MAIN_DIR/weight_bwd_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_bwd_noise --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/act_bwd_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_bwd_noise --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/val_noise_pc --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.val_noise --network_type pc

# Predictive Coding Quant Sweeps
# python3 sweep.py --result_dir $MAIN_DIR/weight_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_bits --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/act_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_bits --network_type pc
python3 sweep.py --result_dir $MAIN_DIR/err_inpt_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_inpt_bits --network_type pc
python3 sweep.py  --result_dir $MAIN_DIR/err_weight_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_weight_bits --network_type pc

# python3 sweep.py --result_dir $MAIN_DIR/weight_bwd_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_bwd_bits --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/act_bwd_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_bwd_bits --network_type pc
# python3 sweep.py --result_dir $MAIN_DIR/val_bits_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.val_bits --network_type pc


# # Backpropagation Noise Sweeps
# python3 sweep.py --result_dir $MAIN_DIR/weight_noise_bp --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_noise --network_type bp
# python3 sweep.py --result_dir $MAIN_DIR/act_noise_bp --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_noise --network_type bp
# python3 sweep.py --result_dir $MAIN_DIR/err_inpt_noise_bp --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_inpt_noise --network_type bp
# python3 sweep.py  --result_dir $MAIN_DIR/err_weight_noise_bp --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_weight_noise --network_type bp

# python3 sweep.py --result_dir $MAIN_DIR/weight_bwd_noise_bp --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_bwd_noise --network_type bp
# python3 sweep.py --result_dir $MAIN_DIR/act_bwd_noise_bp --start $START --stop $STOP --step $STEP --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_bwd_noise --network_type bp


# Backpropagation Quant Sweeps
# python3 sweep.py --result_dir $MAIN_DIR/weight_bits_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_bits --network_type bp
# python3 sweep.py --result_dir $MAIN_DIR/act_bits_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_bits --network_type bp
python3 sweep.py --result_dir $MAIN_DIR/err_inpt_bits_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_inpt_bits --network_type bp
python3 sweep.py  --result_dir $MAIN_DIR/err_weight_bits_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.err_weight_bits --network_type bp

# python3 sweep.py --result_dir $MAIN_DIR/weight_bwd_bits_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter quant.weight_bwd_bits --network_type bp
# python3 sweep.py --result_dir $MAIN_DIR/act_bwd_bits_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q  --parameter quant.act_bwd_bits --network_type bp
