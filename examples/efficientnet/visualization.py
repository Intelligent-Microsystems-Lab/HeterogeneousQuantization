import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorboard as tb
from packaging import version
import grpc

import numpy as np
import pandas as pd

# Validation Runs.


#
# New Sweeps after dynamic correction
#


enets_fp32_t2 = {
    4652008: 'oIBswEoqTme2oLlkkhx4Xw',
    5416680: 'lHDJHMe2RamzzWwF4DF1hQ',
    6092072: 'RiESL4q5QGuzCO66fYZ70Q',
    8197096: 'LvwDTyJbRwKlvE77mBtWhw',
    13006568: 'rdXGZbqPTyuOem2n2s7K0Q',
    'params': 32,
    'label': 'EfficientNet-Lites (FP32)',
}


enet0_lr_best = {
    2: '',
    3: '9nwVcMdPRRGJtjDjrZpCeg',
    4: 'kX80UjhnQxWdwEdzQDTNzQ',
    5: 'KMwHB3liQM2WR7jSzE7nNQ',
    6: 'mx3Gs5p4T0S9KSBXeFqAvg',
    7: 'yZQjrnLBS9iG3DFeUzoaPg',
    8: 'vq1yDQlLSgSbNH0he3QE5A',
    'params': 4652008,
    'label': 'EfficientNet-Lite0 (3-8 Bits)',
}


# Competitor Performance.
competitors = {
    'pact_resnet50': {
        # https://arxiv.org/pdf/1805.06085.pdf
        'eval_err': np.array([1 - 0.722, 1 - 0.753, 1 - 0.765,
                              1 - 0.767]) * 100,
        'size_mb': np.array([2, 3, 4, 5]) * 25503912 / 8_000_000 + 0.21248000,
        'name': 'PACT ResNet50',
        'alpha': .25,
        # no first and last layer quant
    },
    'pact_resnet18': {
        # https://arxiv.org/pdf/1805.06085.pdf
        'eval_err': np.array([1 - 0.644, 1 - 0.681, 1 - 0.692,
                              1 - 0.698]) * 100,
        'size_mb': np.array([2, 3, 4, 5]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'PACT ResNet18',
        'alpha': .25,
        # no first and last layer quant
    },

    'pact_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        'size_mb': np.array([4, 5, 6]) * 3472041 / 8_000_000 + 0.13644800,
        'name': 'PACT MobileNetV2',
        'alpha': .25,
    },

    'dsq_resnet18': {
        # https://arxiv.org/abs/1908.05033
        'eval_err': np.array([1 - 0.6517, 1 - 0.6866, 1 - 0.6956]) * 100,
        'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'DSQ ResNet18',
        'alpha': .25,
    },

    'lsq_resnet18': {
        # https://arxiv.org/abs/1902.08153
        'eval_err': np.array([1 - 0.676, 1 - 0.702, 1 - 0.711,
                              1 - 0.711]) * 100,
        'size_mb': np.array([2, 3, 4, 8]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'LSQ ResNet18',
        'alpha': .25,
    },

    'lsqp_resnet18': {
        # https://arxiv.org/abs/2004.09576
        'eval_err': np.array([1 - 0.668, 1 - 0.694, 1 - 0.708]) * 100,
        'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'LSQ+ ResNet18',
        'alpha': .25,
    },

    'lsqp_enet0': {
        # this is efficient-b0 not lite (!)
        # https://arxiv.org/abs/2004.09576
        'eval_err': np.array([1 - 0.491, 1 - 0.699, 1 - 0.738]) * 100,
        # number might be incorrect
        'size_mb': np.array([2, 3, 4]) * 5246532 / 8_000_000 + 0.16806400,
        'name': 'LSQ+ EfficientNet-B0',
        'alpha': .25,
    },

    'ewgs_resnet18': {
        # https://arxiv.org/abs/2104.00903
        'eval_err': np.array([1 - 0.553, 1 - 0.67, 1 - 0.697,
                              1 - 0.706]) * 100,
        'size_mb': np.array([1, 2, 3, 4]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'EWGS ResNet18',
        'alpha': .25,
    },


    'psgd_resnet18': {
        # https://arxiv.org/pdf/2005.11035.pdf
        # 'eval_err': np.array([1 - 0.7013, 1 - 0.6951, 1 - 0.6345]) * 100,
        # 'size_mb': np.array([8, 6, 4]) * 11679912 / 8_000_000,

        'eval_err': np.array([1 - 0.6345, 1 - 0.6951, 1 - 0.7013,]) * 100,
        'size_mb': np.array([4, 6, 8]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'PSGD ResNet18',
        'alpha': .25,
    },

    'ewgs_resnet34': {
        # https://arxiv.org/abs/2104.00903
        'eval_err': np.array([1 - 0.615, 1 - 0.714, 1 - 0.733,
                              1 - 0.739]) * 100,
        'size_mb': np.array([1, 2, 3, 4]) * 21780648 / 8_000_000 + 0.06809600,
        'name': 'EWGS ResNet34',
        'alpha': .25,
    },

    'qil_resnet18': {
        # We did not quantize the first and the last layers as was done in
        # https://arxiv.org/abs/1808.05779
        # 'eval_err': np.array([1 - 0.704, 1 - 0.701, 1 - 0.692,
        #                       1 - 0.657]) * 100,
        # 'size_mb': np.array([5, 4, 3, 2]) * 11679912 / 8_000_000,


        'eval_err': np.array([1 - 0.657,  1 - 0.692, 1 - 0.701, 1 - 0.704,]) * 100,
        'size_mb': np.array([2, 3, 4, 5]) * 11679912 / 8_000_000 + 0.03840000,
        'name': 'QIL ResNet18*',
        'alpha': .25,
        # no first and last layer quant
    },

    'qil_resnet34': {
        # We did not quantize the first and the last layers as was done in
        # https://arxiv.org/abs/1808.05779
        # 'eval_err': np.array([1 - 0.737, 1 - 0.737, 1 - 0.731,
        #                       1 - 0.706]) * 100,
        # 'size_mb': np.array([5, 4, 3, 2]) * 25557032 / 8_000_000,


        'eval_err': np.array([1 - 0.706, 1 - 0.731, 1 - 0.737, 1 - 0.737,]) * 100,
        'size_mb': np.array([2, 3, 4, 5]) * 21780648 / 8_000_000 + 0.06809600,
        'name': 'QIL ResNet34*',
        'alpha': .25,
        # no first and last layer quant
    },

    'profit_mobilev1': {
        # https://arxiv.org/pdf/2008.04693.pdf
        # 'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        # 'size_mb': np.array([8, 5, 4]) * 3300000 / 8_000_000,


        'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125, ]) * 100,
        'size_mb': np.array([4, 5, 8]) * 4211113 / 8_000_000 + 0.08755200,
        'name': 'PROFIT MobileNetV1',
        'alpha': .25,
    },

    'profit_mobilev2': {
        # https://arxiv.org/pdf/2008.04693.pdf
        # 'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        # 'size_mb': np.array([8, 5, 4]) * 3300000 / 8_000_000,


        'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        'size_mb': np.array([4, 5, 8]) * 3506153 / 8_000_000 + 0.13644800,
        'name': 'PROFIT MobileNetV2',
        'alpha': .25,
    },

    'profit_mobilev3': {
        # https://arxiv.org/pdf/2008.04693.pdf
        # 'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        # 'size_mb': np.array([8, 5, 4]) * 3300000 / 8_000_000,


        'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125,]) * 100,
        'size_mb': np.array([4, 5, 8]) * 5459913 / 8_000_000 + 0.09760000,
        'name': 'PROFIT MobileNetV3',
        'alpha': .25,
    },

    'profit_mnas': {
        # https://arxiv.org/pdf/2008.04693.pdf
        # 'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        # 'size_mb': np.array([8, 5, 4]) * 3300000 / 8_000_000,


        'eval_err': np.array([1 - 0.6139, 1 - 0.6884,  1 - 0.7125]) * 100,
        'size_mb': np.array([4, 5, 8]) * 3853550 / 8_000_000 + 0.13395200,
        'name': 'PROFIT MNasNet-A1',
        'alpha': .25,
    },

}

competitors_bigger_act = {
    'hawqv2_squeeze': {
        # activation bits 8 (uniform)
        # https://arxiv.org/abs/1911.03852
        'eval_err': [1 - 0.6838],
        'size_mb': np.array([1.07]),
        'name': 'HAWQ-V2 SqueezeNext',
        'alpha': 1.,
    },

    # to big to be relevant
    # 'hawqv2_inceptionv3': {
    #     # https://arxiv.org/abs/1911.03852
    #     'eval_err': [1 - 0.7568],
    #     'size_mb': np.array([7.57]),
    #     'name': 'HAWQ-V2 Inception-V3',
    #     'alpha': 1.,
    # },

    'mixed_resnet18': {
        # activation budget 380 KB against 
        # https://arxiv.org/abs/1905.11452
        'eval_err': [0.2992],
        'size_mb': np.array([5.4]),
        'name': 'Mixed Precision DNNs ResNet18',
        'alpha': .15,
    },

    'mixed_mobilev2': {
        # activation budget 570 KB against 
        # https://arxiv.org/abs/1905.11452
        'eval_err': [0.3026],
        'size_mb': np.array([1.55]),
        'name': 'Mixed Precision DNNs MobileNetV2',
        'alpha': .15,
    },

    'haq_mobilev2': {
        # 
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': [1 - 0.6675, 1 - 0.7090, 1 - 0.7147],
        'size_mb': np.array([.95, 1.38, 1.79]),
        'name': 'HAQ MobileNetV2',
        'alpha': 1.,
    },

    'haq_resnet50': {
        # not sure what weight budget is ...
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': [1 - 0.7063, 1 - 0.7530, 1 - 0.7614],
        'size_mb': np.array([6.30, 9.22, 12.14]),
        'name': 'HAQ ResNet50',
        'alpha': 1.,
    },
}

sur_grads_tb = {"STE": "g3EVmBo2Q46JQlyLfzzfkg",
                "Gaussian": "kg86vUopRXWjfVZ9mY1DwQ",
                "Uniform": "7czMCk7rQN2t8Hzc5xkNRA",
                "PSGD": "9xzZuWVxQ3GUOqBzBD2i7g",
                "EWGS": "x4O1lzXmRbOyAxV7JtoE1g",
                "Tanh": "fExUVAqnQMWlA2uqbPLN8A",
                "InvTanh": "NxmL7DdIREOOWiJVGJB1Qg",
                "Acos": "gvma2fO9RAOrrPLNmHVuwg",
                }

sur_grads = ["STE,Gaussian,Uniform,PSGD,EWGS,Tanh,InvTanh,Acos",
             "0.65640,0.65800,0.66260,0.65710,0.66550,0.66620,0.67090,0.65770",
             "0.65540,0.65330,0.66290,0.65560,0.66340,0.65850,0.66030,0.65740",
             "0.66110,0.65610,0.66190,0.65410,0.66050,0.65610,0.66150,0.65560",
             "0.65430,0.66830,0.66050,0.65900,0.66570,0.66310,0.66490,0.66100",
             "0.65790,0.66050,0.65840,0.66310,0.66420,0.66460,0.65870,0.65590",
             "0.65490,0.66100,0.65760,0.66290,0.66260,0.66130,0.65760,0.65800",
             "0.65710,0.65620,0.65720,0.66490,0.66550,0.66500,0.66990,0.65560",
             "0.66110,0.65250,0.66520,0.65590,0.66650,0.66020,0.66290,0.66100",
             "0.66290,0.65950,0.65220,0.66670,0.65970,0.66130,0.66570,0.64960",
             "0.65840,0.65450,0.65840,0.66260,0.65870,0.66100,0.66230,0.65890",
             "0.66020,0.65930,0.66020,0.66080,0.66390,0.65540,0.66680,0.65510",
             "0.65790,0.66440,0.65330,0.66390,0.66780,0.65590,0.66180,0.65950",
             "0.65850,0.66260,0.65900,0.66190,0.66280,0.66370,0.66260,0.65540",
             "0.65710,0.65840,0.65890,0.65820,0.66680,0.66410,0.66470,0.65930",
             "0.65200,0.66420,0.66240,0.65690,0.66650,0.66000,0.66080,0.66440",
             "0.65360,0.65760,0.65970,0.65360,0.65790,0.66260,0.66160,0.65850",
             "0.65330,0.66050,0.65970,0.66210,0.66290,0.66290,0.66110,0.65150",
             "0.65660,0.66080,0.65840,0.66570,0.66630,0.65870,0.66210,0.65890",
             "0.65820,0.65710,0.65450,0.65760,0.65530,0.65980,0.66520,0.66060",
             "0.66130,0.65070,0.65660,0.65840,0.67150,0.66540,0.66390,0.65610",
             ]


sur_grads_mixed = ["STE_PSGD,STE_EWGS,STE_InvTanh,PSGD_STE,EWGS_STE,InvTanh_STE,Acos_STE,InvTanh_EWGS,PSGD_InvTanh,EWGS_InvTanh",
"0.6605,0.6577,0.6589,0.6554,0.6673,0.6628,0.6585,0.6582,0.6649,0.6619",
"0.6548,0.6519,0.6564,0.6574,0.6562,0.6616,0.6582,0.6616,0.6569,0.6654",
"0.6621,0.6584,0.6598,0.6611,0.6657,0.6642,0.6589,0.6639,0.6634,0.6657",
"0.6585,0.6605,0.6562,0.6646,0.6584,0.6618,0.6528,0.6649,0.6593,0.6606",
"0.6543,0.654,0.6574,0.6561,0.6628,0.6629,0.6623,0.6618,0.6572,0.6593",
"0.6561,0.6582,0.6564,0.6598,0.6602,0.6603,0.6592,0.6595,0.6587,0.6637",
"0.6587,0.6628,0.6576,0.6572,0.6623,0.6574,0.6618,0.6676,0.6611,0.6616",
"0.654,0.6585,0.6589,0.661,0.6608,0.6611,0.6611,0.6618,0.6592,0.6621",
"0.6623,0.6592,0.6615,0.6584,0.672,0.6592,0.6556,0.6619,0.6593,0.6672",
"0.6593,0.6562,0.6608,0.6561,0.6605,0.6634,0.6527,0.6593,0.6634,0.6646",
"0.6637,0.6582,0.6506,0.658,0.6649,0.6646,0.6608,0.6683,0.6597,0.6649",
"0.6571,0.6562,0.659,0.6556,0.6642,0.6659,0.6587,0.6589,0.6576,0.6701",
"0.6576,0.6615,0.6602,0.6589,0.6598,0.6615,0.6624,0.6647,0.6628,0.6642",
"0.659,0.6624,,0.6593,0.6624,0.6657,0.6613,0.6628,,",
"0.6556,0.6605,,0.659,0.6636,0.6646,0.659,0.6662,,",
",,,0.6642,0.6626,0.6613,0.6585,,,",
",,,,,,,,,",
",,,,,,,,,",
",,,,,,,,,",
",,,,,,,,,",]

def get_times_rel_ste():

  times_list = []
  names_list = []

  for key, value in sur_grads_tb.items():
    experiment = tb.data.experimental.ExperimentFromDev(value)

    try:
      df = experiment.get_scalars()
    except grpc.RpcError as rpc_error:
      print('Couldn\'t fetch experiment: ' + value + ' got \
          error: ' + str(rpc_error))
      return None

    data = df[df['run'] == 'train']
    times = data[data['tag'] == 'steps_per_second']['value']
    times = times[times > times.mean()]  # discarding first step and eval steps
    times_list.append(1 / times.mean())
    names_list.append(key)

  return times_list# 1 - times_list[0] / np.array(times_list)


def plot_surrogate():
  names = [x.split('_')[0] for x in sur_grads[0].split(',')]
  data = np.stack([x.split(',') for x in sur_grads[1:]])
  y = [float(x) * 100 if x != '' else np.nan for x in data.flatten(order='F')]
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  np_data = np.array(
      [float(x) * 100 if x != '' else np.nan for x in data.flatten('F')
       ]).reshape((-1, 20))
  mu = np.nanmean(np_data, axis=1)
  sigma = np.nanstd(np_data, axis=1)

  font_size = 23

  fig, ax = plt.subplots(figsize=(16.5, 8.5))

  ax.scatter(x / 2, y, marker='x', linewidths=5,
             s=180, color='blue', label='Observations')

  ax.scatter(base_x / 2, mu, marker='_', linewidths=5,
             s=840, color='red', label='Mean')

  ax.scatter(base_x / 2, mu + sigma, marker='_', linewidths=5,
             s=840, color='green', label='Std. Dev.')

  ax.scatter(base_x / 2, mu - sigma, marker='_',
             linewidths=5, s=840, color='green')

  plt.xticks(base_x / 2 + .1, names, rotation='horizontal')

  handles, labels = ax.get_legend_handles_labels()
  handles.append(mpatches.Patch(color='m', label='Compute Overhead'))

  plt.legend(
      handles=handles,
      bbox_to_anchor=(0.5, 1.2),
      loc="upper center",
      ncol=4,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )

  # times = get_times_rel_ste()
  times = np.array([0.00000000e+00, 4.57459557e-01, 4.45819267e-01,
                    2.40027758e-04, 1.25836929e-02, 1.25226535e-01,
                    1.49267876e-01, 1.08655595e-01])

  ax2 = ax.twinx()
  ax2.bar(base_x / 2 + .2, times * 100, width=.1,
          color='m', edgecolor='black', linewidth=3.)
  ax2.set_yscale('log')

  ax.spines["top"].set_visible(False)
  # ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax2.spines["top"].set_visible(False)
  # ax2.spines["right"].set_visible(False)

  ax2.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax2.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(5)

  for tick in ax2.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax2.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  [label.set_fontweight('bold') for label in ax2.get_yticklabels()]

  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  ax2.set_ylabel("Compute Overhead w.r.t. STE (log %)",
                 fontsize=font_size, fontweight='bold')
  plt.tight_layout()
  plt.savefig('figures/surrogate_grads.png')
  plt.close()


def get_best_eval(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)

  try:
    df = experiment.get_scalars()
  except grpc.RpcError as rpc_error:
    print('Couldn\'t fetch experiment: ' + experiment_id + ' got \
        error: ' + str(rpc_error))
    return None

  data = df[df['run'] == 'eval']
  return data[data['tag'] == 'accuracy']['value'].max()


def get_best_eval_and_size(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
  try:
    df = experiment.get_scalars()
  except grpc.RpcError as rpc_error:
    print('Couldn\'t fetch experiment: ' + experiment_id + ' got \
        error: ' + str(rpc_error))
    return None, None

  data = df[df['run'] == 'eval']
  max_eval = data[data['tag'] == 'accuracy']['value'].max()
  if len(data[data['value'] == max_eval]) > 1:
    vals_at_step = data[data['step'] == int(
        data[data['value'] == max_eval]['step'].to_list()[0])]
  else:
    vals_at_step = data[data['step'] == int(
        data[data['value'] == max_eval]['step'])]
  size_mb = float(vals_at_step[vals_at_step['tag'] == 'weight_size']['value'])

  return max_eval, size_mb


def plot_line(ax, res_dict):
  x = []
  y = []
  for key, value in res_dict.items():
    if key == 'label':
      label = value
    elif key == 'params':
      pass
    else:
      acc_temp = get_best_eval(value)
      if acc_temp is not None and acc_temp > .15:
        y.append(1 - acc_temp)
        x.append(key * res_dict['params'] / 8_000_000)

  print(res_dict['label'])
  print(x)
  print(y)
  ax.plot(x, y, marker='x', label=label, ms=20, markeredgewidth=5, linewidth=5)



def plot_comparison(name):
  font_size = 23

  fig, ax = plt.subplots(figsize=(24.5, 12.5))

  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  # Competitors.
  for competitor_name, competitor_data in competitors.items():
    ax.plot(competitor_data['size_mb'], competitor_data['eval_err'],
            label=competitor_data['name'],
            marker='.', ms=20, markeredgewidth=5, linewidth=5,
            alpha=competitor_data['alpha'])

  # Our own.
  # [4.652008, 5.41668, 6.092072, 8.197096, 13.006568]
  ax.plot(np.array([4609992, 5363016, 6033976, 8120424, 12894200]) * 8 / 8_000_000 + np.array([0.16806400, 0.21465600, 0.23238400, 0.30668800, 0.44947200]),
          np.array([0.2452799677848816, 0.2303873896598816, 0.2221272587776184,
                    0.2054443359375, 0.1907958984375]) * 100, marker='x',
          label='EfficientNet-Lite0-4 INT8', ms=20, markeredgewidth=5,
          linewidth=5)

  # [1.744503, 2.326004, 2.907505, 3.489006, 4.070507, 4.652008]
  ax.plot(0.5762490000 * np.array([3,4,5,6,7,8]) + 0.16806400,
          np.array([0.36395263671875, 0.2801310420036316, 0.2576497197151184,
                    0.2494099736213684, 0.2458088994026184,
                    0.2452799677848816]) * 100, marker='x',
          label='EfficientNet-Lite0 (3-8 Bits)', ms=20, markeredgewidth=5,
          linewidth=5)



  xv = 0.577 * np.array([2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0]) + 0.16806400
  yv = 100 - np.array([0.5492, 0.5864, 0.6286, 0.6758,
                       0.6888, 0.6891, 0.7135, 0.7241]) * 100
  ax.plot(xv, yv, marker='x', label="Mixed EfficientNet0",
          ms=20, markeredgewidth=5, linewidth=5, color='red')


  ax.set_xscale('log')
  plt.xticks([1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
             '1.5', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
  ax.set_xlabel("Network Size (MB)", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("Eval Error (%)", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(1., 1.),
      loc="upper left",
      ncol=1,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(name)
  plt.close()


# not used, runtime difference to STE looks odd.
sur_grads_mixed_tb = {
'STE_PSGD': '2c6z1WUlSyGfxbkNenD9cA',
'STE_EWGS': 'PI0wmaaNSZqCNfXqsqSk7A',
'STE_InvTanh': 'xdQtJvCeRl6HavRpCjsbeg',
'PSGD_STE': 'kJBi3rklT4uEnzAzESATqg',
'EWGS_STE': 'A5OrYLJNS7SybC1FLjCWUQ',
'InvTanh_STE': 'PQFx6n8FQECgVzLVV5dMGw',
'Acos_STE': 'wF4mfk4lQSq14OHGWjJPlA',
'InvTanh_EWGS': '2k1wt76LTvimNWo07GtQlQ',
'PSGD_InvTanh': 'qj3Eow1uRxWWqaz0gKXW6Q',
'EWGS_InvTanh': 'vJyM25oIRbGcKSDueA3dAg',
}


def plot_surrogate_mix():

  names = sur_grads_mixed[0].split(',')  # [x.split('_')[0] for x in sur_grads_mixed[0].split(',')]
  names = ['W: ' + x.split('_')[0] + ' A: ' + x.split('_')[1] for x in names]
  data = np.stack([x.split(',') for x in sur_grads_mixed[1:]])
  y = [float(x) * 100 if x != '' else np.nan for x in data.flatten(order='F')]
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  np_data = np.array(
      [float(x) * 100 if x != '' else np.nan for x in data.flatten('F')
       ]).reshape((-1, 20))
  mu = np.nanmean(np_data, axis=1)
  sigma = np.nanstd(np_data, axis=1)

  font_size = 23

  fig, ax = plt.subplots(figsize=(16.5, 11.5))



  ax.scatter(x / 4, y, marker='x', linewidths=5,
             s=180, color='blue', label='Observations')

  ax.scatter(base_x / 4, mu, marker='_', linewidths=5,
             s=840, color='red', label='Mean')

  ax.scatter(base_x / 4, mu + sigma, marker='_', linewidths=5,
             s=840, color='green', label='Std. Dev.')

  ax.scatter(base_x / 4, mu - sigma, marker='_',
             linewidths=5, s=840, color='green')

  ax.axhline(y=65.741000, color='orange', linestyle='--', label = 'STE', linewidth=5)

  
  plt.xticks(base_x / 4, names, rotation=45, horizontalalignment='right')


  plt.legend(
      bbox_to_anchor=(0.5, 1.2),
      loc="upper center",
      ncol=4,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )

 
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')


  # [label.set_fontweight('bold') for label in ax.get_yticklabels()]

  # plt.xticks(rotation = 45)
  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  #ax2.set_ylabel("Compute Overhead w.r.t. STE (log %)",
  #               fontsize=font_size, fontweight='bold')
  plt.tight_layout()
  plt.savefig('figures/surrogate_grads_mixed.png')
  plt.close()

if __name__ == '__main__':
  major_ver, minor_ver, _ = version.parse(tb.__version__).release
  assert major_ver >= 2 and minor_ver >= 3, \
      "This notebook requires TensorBoard 2.3 or later."
  print("TensorBoard version: ", tb.__version__)

  plot_surrogate()
  plot_surrogate_mix()
  plot_comparison('figures/overview.png')
  # plot_comparison_bigger_act('figures/overview_bigger_act.png')

