import subprocess
import re
import numpy as np

base_config_name = 'efficientnet/configs/efficientnet-lite0_w2a2.py'
init_methods = ['partial(max_init)', 'partial(double_mean_init)', 'partial(gaussian_init)', 'partial(percentile_init,perc=99.9)',
                'partial(percentile_init,perc=99.99)', 'partial(percentile_init,perc=99.999)', 'partial(percentile_init,perc=99.9999)']
config_dir = 'efficientnet/configs/init_ablation_w2a2/'

name_rm_chars = [',', '=', '.', ')']
num_run_scripts = 3

with open(base_config_name) as f:
  base_config = f.readlines()

# rm dir when exists
subprocess.run(["rm", "-rf", config_dir])

# create dir
subprocess.run(["mkdir", config_dir])


config_list = []

for w_init in init_methods:
  for a_init in init_methods:

    new_conf_name = config_dir + '/' + base_config_name.split('/')[-1].split('.')[0] + '_w_init_' + re.sub(
        '[,=.)]', '', w_init.split('(')[1]) + '_a_init_' + re.sub('[,=.)]', '', a_init.split('(')[1]) + '.py'

    print(new_conf_name)
    config_list.append(new_conf_name)

    with open(new_conf_name, 'w') as f:
      for line in base_config:
        if 'from quant import' in line:
          if w_init == a_init:
            f.write(line[:-1] + ', ' + w_init.split('(')
                    [1].split(',')[0].split(')')[0] + '\n')
          else:
            f.write(line[:-1] + ', ' + w_init.split('(')[1].split(',')[0].split(')')
                    [0] + ', ' + a_init.split('(')[1].split(',')[0].split(')')[0] + '\n')
        elif '.act =' in line or '.average =' in line:
          f.write(line[:-2] + ', init_fn = ' + a_init + ')\n')
        elif '.weight =' in line or '.bias =' in line:
          f.write(line[:-2] + ', init_fn = ' + w_init + ')\n')
        else:
          f.write(line)

i = 0
for idx, conf in enumerate(config_list):
  if (idx % np.ceil(len(config_list) / num_run_scripts)) == 0:
    i = i + 1
    with open(config_dir + '/run_init_ablation_' + str(i) + '.sh', 'w') as f:
      f.write('')

  with open(config_dir + '/run_init_ablation_' + str(i) + '.sh', 'a+') as f:
    f.write('python3 train.py --workdir=/home/clemens/'
            + conf.split('/')[-1][:-3] + ' --config=' + conf + '\n')
