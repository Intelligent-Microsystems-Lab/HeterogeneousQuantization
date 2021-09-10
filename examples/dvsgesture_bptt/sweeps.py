import subprocess

param_sweep = {
    # "init_scale_s": [1, 0.5, 0.1, 0.05, 0.001],
    # "learning_rate": [0.5, 10e-2, 10e-3, 10e-4, 10e-5],
    # "update_freq": [1, 10, 50, 200, 500],
    # "grad_clip": [1, 5, 50, 100],
}

for key, sweep_list in param_sweep.items():
  for param_val in sweep_list:
    process = subprocess.Popen(
        'XLA_FLAGS="--xla_tpu_detect_nan=true" python3 train.py --'
        + key
        + " "
        + str(param_val),
        shell=True,
    )
    process.wait()
    print(process.returncode)
