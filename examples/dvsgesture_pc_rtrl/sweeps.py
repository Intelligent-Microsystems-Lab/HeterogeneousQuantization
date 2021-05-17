import subprocess
import numpy as np

param_sweep = {
    'init_scale_s':[1, 0.5, .1, .05, .001],
    'learning_rate':[.5, 10e-2, 10e-3, 10e-4, 10e-5, 10e-6],
    'update_freq':[1, 5, 10, 50, 200, 500],
    'grad_clip':[1, 5, 50, 100],
}

for key, sweep_list in param_sweep.items():
    for param_val in sweep_list:
        process = subprocess.Popen('XLA_FLAGS="--xla_tpu_detect_nan=true" python3 train.py --' + key + ' ' + str(param_val), shell=True) # , stdout=subprocess.PIPE

        process.wait()
        print(process.returncode)