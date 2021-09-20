#!/bin/bash
module load python

# Several hyperparameter sweeps on Notre Dame CRC cluster.
MAIN_DIR="/afs/crc.nd.edu/user/c/cschaef6/cifar10_hyper"
USER="cschaef6"
GPU_Q="gpu@@joshi"
TRIALS="3"




mkdir $MAIN_DIR

LIST="16,32,64,128"

python3 sweep.py --result_dir $MAIN_DIR/batch_size_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter batch_size --network_type pc

python3 sweep.py --result_dir $MAIN_DIR/batch_size_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter batch_size --network_type bp

LIST="0.0001,0.001,0.01,0.1"

python3 sweep.py --result_dir $MAIN_DIR/learning_rate_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter learning_rate --network_type pc

python3 sweep.py --result_dir $MAIN_DIR/learning_rate_bp --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter learning_rate --network_type bp



LIST="0.001,0.01,0.1,0.5"

python3 sweep.py --result_dir $MAIN_DIR/infer_lr_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter infer_lr --network_type pc

LIST="1,20,50,100,200"

python3 sweep.py  --result_dir $MAIN_DIR/infer_steps_pc --list $LIST --user $USER --trials $TRIALS --queue $GPU_Q --parameter infer_steps --network_type pc


