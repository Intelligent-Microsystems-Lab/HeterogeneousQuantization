import subprocess
import re
import numpy as np

base_config_name = 'efficientnet/configs/efficientnet-lite0_w3a3.py'
round_methods = ['round_tanh', 'round_invtanh']
scale_factors = [1.0, .9, .8, .7, .6, .5, .4, .3, .2, 1., 0.]
config_dir = 'efficientnet/configs/alpha_scale_ablation_w3a3'

name_rm_chars = [',', '=', '.', ')']
num_run_scripts = 4

with open(base_config_name) as f:
  base_config = f.readlines()

# rm dir when exists
subprocess.run(["rm", "-rf", config_dir])

# create dir
subprocess.run(["mkdir", config_dir])


config_list = []

for round_fn in round_methods:
  for scale_v in scale_factors:

    new_conf_name = config_dir + '/' + base_config_name.split('/')[2].split(
        '.')[0] + '_' + round_fn + '_alpha_scale_' + str(scale_v) + '.py'

    config_list.append(new_conf_name)

    with open(new_conf_name, 'w') as f:
      for line in base_config:
        if 'from quant import' in line:
          f.write(line[:-1] + ', ' + round_fn + '\n')
        elif '.act =' in line or '.average =' in line or '.weight =' in line or '.bias =' in line:
          f.write(line[:-2] + ', round_fn = partial(' + round_fn +
                  ', alpha_scale = ' + str(scale_v) + ') )\n')
        elif 'g_scale' in line:
          f.write('  config.quant.g_scale = ' +
                  ('5e-3' if round_fn == 'round_tanh' else '1e-2') + '\n')
        else:
          f.write(line)

i = 0
for idx, conf in enumerate(config_list):
  if (idx % np.ceil(len(config_list) / num_run_scripts)) == 0:
    i = i + 1
    with open(config_dir + '/run_round_ablation_' + str(i) + '.sh', 'w') as f:
      f.write('')

  with open(config_dir + '/run_round_ablation_' + str(i) + '.sh', 'a+') as f:
    f.write('python3 train.py --workdir=../../' +
            conf.split('/')[3][:-3] + ' --config=' + conf + '\n')
