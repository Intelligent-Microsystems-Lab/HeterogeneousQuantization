import subprocess
import numpy as np

param_sweep = {
    "init_scale_s": np.arange(),
    #'learning_rate':np.arange(),
    #'update_freq':np.arange(),
    #'grad_clip':np.arange(),
}

for key, sweep_list in param_sweep.items():
    for param_val in sweep_list:
        process = subprocess.Popen(
            'XLA_FLAGS="--xla_tpu_detect_nan=true" python3 train.py  --training_epochs 1 --'
            + key
            + " "
            + str(param_val),
            shell=True,
        )  # , stdout=subprocess.PIPE

        process.wait()
        print(process.returncode)
