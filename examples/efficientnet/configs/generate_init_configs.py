import subprocess

base_config_name = 'configs/efficientnet-lite0_w8a8_static.py'
init_methods = ['partial(max_init)', 'partial(double_mean_init)', 'partial(gaussian_init)', 'partial(entropy_init)', 'partial(percentile_init,perc=99.9)', 'partial(percentile_init, perc=99.99)', 'partial(percentile_init,perc=99.999)', 'partial(percentile_init,perc=99.9999)']
config_dir = 'configs/init_ablation'

name_rm_chars = [',','=','.',')']

with open(base_config_name) as f:
  base_config = f.readlines()

# rm dir when exists
subprocess.run(["rm", "-rf", config_dir])

# create dir
subprocess.run(["mkdir", config_dir])

for w_init in init_methods:
  for a_init in init_methods:


    with open(config_dir +  '/' +base_config_name.split('/')[1].split('.')[0] +  '_w_init_' + w_init + '_a_init_' + a_init + '.py', 'w') as f:
      for line in base_config:
        if 'from quant import' in line:
          f.write(line+ ', ' + w_init)
        print(line)

    import pdb; pdb.set_trace()