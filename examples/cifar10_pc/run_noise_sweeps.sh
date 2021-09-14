#!/bin/bash
module load python
python3 sweep.py --parameter noise_weight --from .0 --to .25 --steps 25 --network_type pc --result_dir cifar10_pc_weight_noise 

