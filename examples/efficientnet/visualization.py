import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import tensorboard as tb
from packaging import version
import grpc
import string
from collections import OrderedDict

import numpy as np
import itertools

# 'pact_resnet50': {
#     # https://arxiv.org/pdf/1805.06085.pdf
#     'eval_err': np.array([1 - 0.722, 1 - 0.753, 1 - 0.765,
#                           1 - 0.767]) * 100,
#     'size_mb': np.array([2, 3, 4, 5
#                          ]) * 25503912 / 8_000_000 + 0.21248000 / 2,
#     'name': 'PACT ResNet50*',
#     'alpha': .25,
#     # no first and last layer quant
# },
# 'pact_resnet18': {
#     # https://arxiv.org/pdf/1805.06085.pdf
#     'eval_err': np.array([1 - 0.644, 1 - 0.681, 1 - 0.692,
#                           1 - 0.698]) * 100,
#     'size_mb': np.array([2, 3, 4, 5
#                          ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
#     'name': 'PACT ResNet18*',
#     'alpha': .25,
#     # no first and last layer quant
# },
# 'lsqp_resnet18': {
#     # https://arxiv.org/abs/2004.09576
#     # they claim to be a natural extension... so also first and last?
#     'eval_err': np.array([1 - 0.668, 1 - 0.694, 1 - 0.708]) * 100,
#     'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000 + 0.03840000/2,
#     'name': 'LSQ+ ResNet18*',
#     'alpha': .25,
# },
# 'ewgs_resnet34': {
#     # https://arxiv.org/abs/2104.00903
#     'eval_err': np.array([1 - 0.615, 1 - 0.714, 1 - 0.733,
#                           1 - 0.739]) * 100,
#     'size_mb': np.array([1, 2, 3, 4
#                          ]) * 21780648 / 8_000_000 + 0.06809600 / 2,
#     'name': 'EWGS ResNet34*',
#     'alpha': .25,
# },
# 'qil_resnet34': {
#     # "We did not quantize the first and the last layers as was done in"
#     # https://arxiv.org/abs/1808.05779
#     'eval_err': np.array([1 - 0.706, 1 - 0.731, 1 - 0.737, 1 - 0.737,
#                           ]) * 100,
#     'size_mb': np.array([2, 3, 4, 5
#                          ]) * 21780648 / 8_000_000 + 0.06809600 / 2,
#     'name': 'QIL ResNet34*',
#     'alpha': .25,
# },
# 'profit_mobilev1': {
#     # https://arxiv.org/pdf/2008.04693.pdf
#     'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125, ]) * 100,
#     'size_mb': np.array([4, 5, 8]) * 4211113 / 8_000_000 + 0.08755200 /2,
#     'name': 'PROFIT MobileNetV1',
#     'alpha': .25,
# },
# 'profit_mobilev3': {
#     # https://arxiv.org/pdf/2008.04693.pdf
#     'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125, ]) * 100,
#     'size_mb': np.array([4, 5, 8]) * 5459913 / 8_000_000 + 0.09760000 /2,
#     'name': 'PROFIT MobileNetV3',
#     'alpha': .25,
# },
# 'profit_mnas': {
#     # https://arxiv.org/pdf/2008.04693.pdf
#     'eval_err': np.array([1 - .72244, 1 - .73378, 1 - .73742]) * 100,
#     'size_mb': np.array([4, 5, 8]) * 3853550 / 8_000_000 + 0.13395200 / 2,
#     'name': 'PROFIT MNasNet-A1',
#     'alpha': .25,
# },
# 'mixed_resnet18': {
#     # activation budget 380 KB against
#     # https://arxiv.org/abs/1905.11452
#     'eval_err': np.array([0.2992]) * 100,
#     'size_mb': np.array([5.4]),
#     'name': 'Mixed ResNet18',
#     'alpha': .25,
# },
# 'haq_resnet50': {
#     # https://arxiv.org/pdf/1811.08886.pdf
#     'eval_err': np.array([1 - 0.7063, 1 - 0.7530, 1 - 0.7614]) * 100,
#     'size_mb': np.array([6.30, 9.22, 12.14]),
#     'name': 'HAQ ResNet50',
#     'alpha': .25,
# },
# to big to be relevant
# 'hawqv2_inceptionv3': {
#     # https://arxiv.org/abs/1911.03852
#     'eval_err': [1 - 0.7568],
#     'size_mb': np.array([7.57]),
#     'name': 'HAWQ-V2 Inception-V3',
#     'alpha': 1.,
# },


# Competitor Performance.
competitors = {
    'pact_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.pdf
        # no first and last layer quant
        'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        'size_mb': np.array([4, 5, 6
                             ]) * (3472041 + 6616672) / 8_000_000 + 0.06822,
        'name': 'PACT MobileNetV2',  # 'PACT MobileNetV2*',
        'alpha': .25,
    },

    'dsq_resnet18': {
        # https://arxiv.org/abs/1908.05033
        'eval_err': np.array([1 - 0.6517, 1 - 0.6866, 1 - 0.6956]) * 100,
        'size_mb': np.array([2, 3, 4
                             ]) * (11157504 + 2032128) / 8_000_000 + 0.0192 \
        + 1.044816 + (512 * 16 / 8_000_000),
        'name': 'DSQ ResNet18',
        'alpha': .25,
    },

    'lsq_resnet18': {
        # https://arxiv.org/abs/1902.08153
        # no first and last layer quant
        'eval_err': np.array([1 - 0.676, 1 - 0.702, 1 - 0.711,
                              1 - 0.711]) * 100,
        'size_mb': np.array([2, 3, 4, 8
                             ]) * (11157504 + 2032128) / 8_000_000 + 0.0192 \
        + 1.044816 + (512 * 16 / 8_000_000),
        'name': 'LSQ ResNet18',
        'alpha': .25,
    },

    'lsqp_enet0': {
        # this is efficient-b0 not lite (!)
        # https://arxiv.org/abs/2004.09576
        # they claim to be a natural extension... so also first and last?
        'eval_err': np.array([1 - 0.491, 1 - 0.699, 1 - 0.738]) * 100,
        # number might be incorrect
        'size_mb': np.array([2, 3, 4]) * (3964668 + 8981504) / 8_000_000 \
        + (1281000 * 16 / 8_000_000) + \
        (864 * 16 / 8_000_000) + (1280 * 16 / 8_000_000),
        'name': 'LSQ+ EfficientNet-B0',
        'alpha': .25,
    },

    'ewgs_resnet18': {
        # https://arxiv.org/abs/2104.00903
        # no first and last layer quant
        'eval_err': np.array([1 - 0.67, 1 - 0.697,
                              1 - 0.706]) * 100,  # 1 - 0.553,
        'size_mb': np.array([2, 3, 4
                             ]) * (11157504 + 2032128) / 8_000_000 + 0.0192 \
        + 1.044816 + (512 * 16 / 8_000_000),
        'name': 'EWGS ResNet18',
        'alpha': .25,
    },


    'psgd_resnet18': {
        # https://arxiv.org/pdf/2005.11035.pdf
        'eval_err': np.array([1 - 0.6345, 1 - 0.6951, 1 - 0.7013, ]) * 100,
        'size_mb': np.array([4, 6, 8]) * (11157504 + 2032128) / 8_000_000 \
        + 0.0192 + 0.522408 + (512 * 8 / 8_000_000),
        'name': 'PBGS ResNet18',
        'alpha': .25,
    },

    'qil_resnet18': {
        # "We did not quantize the first and the last layers as was done in..."
        # https://arxiv.org/abs/1808.05779
        'eval_err': np.array([1 - 0.657, 1 - 0.692, 1 - 0.701, 1 - 0.704,
                              ]) * 100,
        'size_mb': np.array([2, 3, 4, 5
                             ]) * (11157504 + 2032128) / 8_000_000 + 0.0192 \
        + 1.044816 + (512 * 16 / 8_000_000),
        'name': 'QIL ResNet18',
        'alpha': .25,
    },

    'profit_mobilev2': {
        # https://arxiv.org/pdf/2008.04693.pdf
        'eval_err': np.array([1 - .71564, 1 - .72192, 1 - .72352]) * 100,
        'size_mb': np.array([4, 5, 8]) * (3470760 + 6678112) / 8_000_000 \
        + (34112 * 16 / 8_000_000),
        'name': 'PROFIT MobileNetV2',
        'alpha': .25,
    },

    'hawqv2_squeeze': {
        # activation bits 8 (uniform)
        # https://arxiv.org/abs/1911.03852
        'eval_err': np.array([1 - 0.6838]) * 100,
        'size_mb': np.array([1.07]) + 0.369664 + 3.962208,
        'name': 'HAWQ SqueezeNext',
        'alpha': .25,
    },

    'mixed_mobilev2': {
        # activation budget 570 KB against -> be generous 4 bits
        # 6616672 * 4.0
        # https://arxiv.org/abs/1905.11452
        'eval_err': np.array([0.3026]) * 100,
        'size_mb': np.array([1.55]) + 0.068224 + 3.308336,
        'name': 'Mixed MobileNetV2',
        'alpha': .25,
    },

    'haq_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.
        # didnt address batch norm
        # I think they only did weight compression and latency considerations.
        # give them 8 bits for act
        'eval_err': np.array([1 - 0.6675, 1 - 0.7090, 1 - 0.7147]) * 100,
        'size_mb': np.array([.95, 1.38, 1.79]) + 0.068224 + 6.616672,
        'name': 'HAQ MobileNetV2',
        'alpha': .25,
    },

}


sur_grads_tb = {"STE": "g3EVmBo2Q46JQlyLfzzfkg",
                "Gaussian": "kg86vUopRXWjfVZ9mY1DwQ",
                "Uniform": "7czMCk7rQN2t8Hzc5xkNRA",
                "PSGD": "9xzZuWVxQ3GUOqBzBD2i7g",
                "EWGS": "x4O1lzXmRbOyAxV7JtoE1g",
                "Tanh": "fExUVAqnQMWlA2uqbPLN8A",
                "ATanh": "NxmL7DdIREOOWiJVGJB1Qg",
                "Acos": "gvma2fO9RAOrrPLNmHVuwg",
                }


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

  return 1 - times_list[0] / np.array(times_list)


def plot_surrogate():
  data = np.genfromtxt('figures/surrogate_plain.csv',
                       delimiter=',', names=True)
  names = data.dtype.names
  data = np.genfromtxt('figures/surrogate_plain.csv',
                       delimiter=',', skip_header=1) * 100
  y = data.flatten(order='F')
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  font_size = 23
  gen_linewidth = 3
  plt.rc('font', family='Helvetica', weight='bold')
  fig, ax = plt.subplots(figsize=(14.4, 6.5))

  boxprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')
  flierprops = dict(marker='o', linewidth=gen_linewidth,)
  medianprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')
  meanlineprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')

  bp_data = np.stack([y[np.argwhere(x == i)]
                     for i in np.unique(x)])[:, :, 0].transpose()
  ax.boxplot(bp_data, positions=np.unique(x), boxprops=boxprops,
             whiskerprops=boxprops,
             capprops=boxprops, flierprops=flierprops, medianprops=medianprops,
             meanprops=meanlineprops)
  ax.scatter(x, y, marker='.', linewidths=0,
             s=180, alpha=.4, color='blue', label='Observations')

  plt.xticks(base_x + .2, names, rotation='horizontal')

  handles, labels = ax.get_legend_handles_labels()
  handles.append(mpatches.Patch(
      facecolor='m', label='Wall-clock Overhead', edgecolor='black',
      linewidth=3.))

  plt.legend(
      handles=handles,
      bbox_to_anchor=(0., 1.02, 1.0, .1),
      loc="upper center",
      ncol=4,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )

  times = np.array([0.00000000e+00, 4.57459557e-01, 4.45819267e-01,
                    2.40027758e-04, 1.25836929e-02, 1.25226535e-01,
                    1.49267876e-01, 1.08655595e-01])

  ax2 = ax.twinx()
  ax2.bar(base_x + .5, times * 100, width=.1,
          color='m', edgecolor='black', linewidth=gen_linewidth)
  ax2.set_yscale('log')

  ax.spines["top"].set_visible(False)

  ax.xaxis.set_tick_params(width=gen_linewidth, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=gen_linewidth, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(gen_linewidth)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax2.spines["top"].set_visible(False)

  ax2.xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax2.yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(gen_linewidth)

  for tick in ax2.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax2.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  [label.set_fontweight('bold') for label in ax2.get_yticklabels()]

  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  ax2.set_ylabel("Wall-clock Overhead w.r.t. STE (log %)",
                 fontsize=font_size, fontweight='bold')
  plt.tight_layout()
  plt.savefig('figures/surrogate_grads.png', dpi=300)
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
  gen_lw = 3
  mw = 15

  plt.rc('font', family='Helvetica', weight='bold')
  fig, ax = plt.subplots(figsize=(14.4, 8.5))

  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=gen_lw, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=gen_lw, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(gen_lw)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  # Competitors.
  for competitor_name, competitor_data in competitors.items():
    ax.plot(competitor_data['size_mb'], competitor_data['eval_err'],
            label=competitor_data['name'],
            marker='.', ms=mw, markeredgewidth=gen_lw, linewidth=gen_lw,
            alpha=competitor_data['alpha'])

  # gran sur pre train
  ax.plot((0.5762490000 + 0.834532) * np.array([3, 4, 5, 6, 7, 8]) + 0.084032,
          (1 - np.array([0.6678, 0.7269, 0.7425, 0.7466, 0.7497, 0.7505]))
          * 100, marker='.',
          label='EfficientNet-Lite0 3-8 Bits', ms=mw, markeredgewidth=gen_lw,
          linewidth=gen_lw)

  # Our own.

  # squeezenext
  ax.plot(np.array([2.6015712276, 3.4095508423000003, 4.606633788999999,
                    4.8350780030000005]) + 0.369664, [50.421099999999996,
                                                      40.2303,

                                                      36.03920000000001,
                                                      35.6628], marker='x',
          label='SqueezeNext (Ours)', ms=mw, markeredgewidth=gen_lw,
          linewidth=gen_lw)

  # mobilenet
  ax.plot(np.array([2.8214422603, 3.1445559693, 3.411231812,
                    3.4379381110000002, 4.753455323, 4.979867553, 5.551028442,
                    5.694456177]
                   ) + 0.068224, [39.540600000000005, 36.8205,
                                  34.802200000000006, 34.606899999999996,
                                  31.5002, 31.0689, 30.454499999999996,
                                  30.322300000000002],
          marker='x',
          label='MobileNetV2 (Ours)', ms=mw, markeredgewidth=gen_lw,
          linewidth=gen_lw)

  # efficientnet
  ax.plot(np.array([2.925111084, 2.925306152, 3.1460351570000005, 3.9969198,
                    5.373107422, 5.788132813, 5.932807129]) + 0.084032,
          [51.7965, 51.6317, 47.127300000000005, 33.247899999999994,
           27.337599999999995, 26.867700000000006,
           26.837200000000006], marker='x',
          label='EfficientNet-Lite0 (Ours)', ms=mw, markeredgewidth=gen_lw,
          linewidth=gen_lw)

  ax.set_xscale('log')
  plt.xticks([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [
      '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
  ax.set_xlabel("Sum Parameter Size + Sum Activation Feauture Maps Size (MB)",
                fontsize=font_size, fontweight='bold')
  ax.set_ylabel("Eval Error (%)", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(-.05, 1.02, 1.05, 0.2),
      loc="lower left",
      ncol=3,
      mode="expand",
      borderaxespad=0,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(name, dpi=300)
  plt.close()


# not used, runtime difference to STE looks odd.
sur_grads_mixed_tb = {
    'STE_PSGD': '2c6z1WUlSyGfxbkNenD9cA',
    'STE_EWGS': 'PI0wmaaNSZqCNfXqsqSk7A',
    'STE_ATanh': 'xdQtJvCeRl6HavRpCjsbeg',
    'PSGD_STE': 'kJBi3rklT4uEnzAzESATqg',
    'EWGS_STE': 'A5OrYLJNS7SybC1FLjCWUQ',
    'ATanh_STE': 'PQFx6n8FQECgVzLVV5dMGw',
    'Acos_STE': 'wF4mfk4lQSq14OHGWjJPlA',
    'ATanh_EWGS': '2k1wt76LTvimNWo07GtQlQ',
    'PSGD_ATanh': 'qj3Eow1uRxWWqaz0gKXW6Q',
    'EWGS_ATanh': 'vJyM25oIRbGcKSDueA3dAg',
}


def plot_meth_sur():
  data = np.genfromtxt('figures/surrogate_mixed.csv',
                       delimiter=',', names=True)
  names = data.dtype.names
  names = ['W: ' + x.split('_')[0] + ' A: ' + x.split('_')[1] for x in names]
  names = string.ascii_lowercase[:len(names)]
  data = np.genfromtxt('figures/surrogate_mixed.csv',
                       delimiter=',', skip_header=1) * 100
  y = data.flatten(order='F')
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  font_size = 23
  gen_linewidth = 3

  plt.rc('font', family='Helvetica', weight='bold')

  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14.4, 6.5))

  textst1 = 'Configurations:\n\na)\nb)\nc)\nd)\ne)\nf)\ng)\nh)\ni)\nj)'
  textst2 = '\nWgt.\nSTE\nSTE\nSTE\nPBGS\nEWGS\nATanh\nAcos\nATanh\nPBGS\nEWGS'
  textst3 = '\nAct.\nPBGS\nEWGS\nATanh\nSTE\nSTE\nSTE\nSTE\nEWGS\nATanh\nATanh'

  plt.text(.88, 0.25, textst1, fontsize=20,
           transform=plt.gcf().transFigure, linespacing=1.5)
  plt.text(.90, 0.25, textst2, fontsize=20,
           transform=plt.gcf().transFigure, linespacing=1.5)
  plt.text(.965, 0.25, textst3, fontsize=20,
           transform=plt.gcf().transFigure, linespacing=1.5)

  ax[1].axhline(y=65.741000, color='orange',
                linestyle='--', label='STE', linewidth=gen_linewidth)

  ax[1].axhline(y=66.37000, color='purple',
                linestyle='--', label='EWGS', linewidth=gen_linewidth)

  boxprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')
  flierprops = dict(marker='o', linewidth=gen_linewidth,)
  medianprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')
  meanlineprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')

  bp_data = np.stack([y[np.argwhere(x == i)]
                     for i in np.unique(x)])[:, :, 0].transpose()
  ax[1].boxplot(bp_data, positions=np.unique(x), boxprops=boxprops,
                whiskerprops=boxprops,
                capprops=boxprops, flierprops=flierprops,
                medianprops=medianprops, meanprops=meanlineprops)
  ax[1].scatter(x, y, marker='.', linewidths=0,
                s=180, alpha=.4, color='blue', label='Obs.')

  ax[1].set_xticks(base_x, names, rotation=0, horizontalalignment='center')

  ax[1].spines["top"].set_visible(False)
  ax[1].spines["right"].set_visible(False)

  ax[1].xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax[1].yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax[1].spines[axis].set_linewidth(gen_linewidth)

  for tick in ax[1].xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax[1].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax[1].set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  ax[1].set_xlabel("Configuration", fontsize=font_size, fontweight='bold')

  ax[0].spines["top"].set_visible(False)
  ax[0].spines["right"].set_visible(False)

  ax[0].xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax[0].yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(gen_linewidth)

  for tick in ax[0].xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax[0].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  x = np.arange(-1, 1, .001)

  ax[0].plot(x, x * 0, linewidth=gen_linewidth,
             label='STE', linestyle='--', color='orange')
  ax[0].plot(x[::100], np.abs(x)[::100], linewidth=gen_linewidth,
             label='PBGS', marker='x', ms=8, markeredgewidth=gen_linewidth,
             color='g')
  ax[0].plot(x, x, linewidth=gen_linewidth, label='EWGS',
             linestyle='--', color='purple')
  ax[0].plot(x, np.sin(np.pi * (.5 * x - np.round(.5 * x))),
             linewidth=gen_linewidth, label='ACos', color='b')
  ax[0].plot(x, np.tanh(x * 4), linewidth=gen_linewidth,
             label='Tanh', color='r')
  ax[0].plot(x, np.arctanh(x / 1.05) / 1.85,
             linewidth=gen_linewidth, label='ATanh', color='k')

  ax[0].set_xticks([-1, 0, 1], [
      r'$QP\,$-$\,.5d$', r'$QP$', r'$QP+.5d$'])
  ax[0].set_yticks([-1, 0, 1], [
      r'$1\,$-$\,\delta$', r'$1$', r'$1+\delta$'])
  ax[0].set_xlabel("x", fontsize=font_size, fontweight='bold')
  ax[0].set_ylabel("df/dx", fontsize=font_size, fontweight='bold')

  handles, labels = ax[0].get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  fig.legend(
      by_label.values(), by_label.keys(),
      bbox_to_anchor=(0.15, .95, .6, 0.2),
      loc="lower left",
      ncol=3,
      mode="expand",
      borderaxespad=0,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )

  plt.tight_layout()
  plt.savefig('figures/meth_sur.png', dpi=300, bbox_inches='tight')
  plt.close()


def flip(items, ncol):
  return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def plot_surrogate_24():

  font_size = 23
  gen_linewidth = 3
  plt.rc('font', family='Helvetica', weight='bold')

  fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(
      14.4, 6.5), gridspec_kw={'width_ratios': [20, 20, 1]})

  ax[2].get_xaxis().set_visible(False)
  ax[2].get_yaxis().set_visible(False)
  ax[2].spines["top"].set_visible(False)
  ax[2].spines["right"].set_visible(False)
  ax[2].spines["bottom"].set_visible(False)
  ax[2].spines["left"].set_visible(False)

  textstr1 = 'Configurations\n\na)\nb)\nc)\nd)\ne)\nf)'
  textstr2 = '\nWgt.\nSTE\nEWGS\nEWGS\nATanh\nEWGS\nATanh'
  textstr3 = 'Act.\nSTE\nEWGS\nATanh\nEWGS\nSTE\nSTE'

  base_x = .82
  base_y = .27
  plt.text(base_x, base_y, textstr1, fontsize=20,
           transform=plt.gcf().transFigure, linespacing=1.5)
  plt.text(base_x + .025, base_y, textstr2, fontsize=20,
           transform=plt.gcf().transFigure, linespacing=1.5)
  plt.text(base_x + .1, base_y, textstr3, fontsize=20,
           transform=plt.gcf().transFigure, linespacing=1.5)

  ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  boxprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')
  flierprops = dict(marker='o', linewidth=gen_linewidth,)
  medianprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')
  meanlineprops = dict(linestyle='-', linewidth=gen_linewidth, color='k')

  data = np.genfromtxt('figures/surrogate_plain' + str(2) + '.csv',
                       delimiter=',', names=True)
  names = data.dtype.names
  data = np.genfromtxt('figures/surrogate_plain' + str(2) + '.csv',
                       delimiter=',', skip_header=1) * 100
  y = data.flatten(order='F')
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  bp_data = np.stack([y[np.argwhere(x == i)]
                     for i in np.unique(x)])[:, :, 0].transpose()
  ax[0].boxplot(bp_data, positions=np.unique(x), boxprops=boxprops,
                whiskerprops=boxprops,
                capprops=boxprops, flierprops=flierprops,
                medianprops=medianprops, meanprops=meanlineprops)
  ax[0].scatter(x, y, marker='.', linewidths=0,
                s=180, alpha=.4, color='blue', label='2 Bits')

  data = np.genfromtxt('figures/surrogate_plain' + str(4) + '.csv',
                       delimiter=',', names=True)
  names = data.dtype.names
  data = np.genfromtxt('figures/surrogate_plain' + str(4) + '.csv',
                       delimiter=',', skip_header=1) * 100
  y = data.flatten(order='F')
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  bp_data = np.stack([y[np.argwhere(x == i)]
                     for i in np.unique(x)])[:, :, 0].transpose()
  ax[1].boxplot(bp_data, positions=np.unique(x), boxprops=boxprops,
                whiskerprops=boxprops,
                capprops=boxprops, flierprops=flierprops,
                medianprops=medianprops, meanprops=meanlineprops)
  ax[1].scatter(x, y, marker='.', linewidths=0,
                s=180, alpha=.4, color='red', label='4 Bits')

  ax[0].spines["top"].set_visible(False)
  ax[0].spines["right"].set_visible(False)

  ax[0].xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax[0].yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(gen_linewidth)

  for tick in ax[0].xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax[0].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax[0].set_ylabel("Eval Accuracy (%)",
                   fontsize=font_size, fontweight='bold')
  ax[1].set_xlabel("Configurations",
                   fontsize=font_size, fontweight='bold')
  ax[0].set_xlabel("Configurations",
                   fontsize=font_size, fontweight='bold')

  ax[1].spines["top"].set_visible(False)
  ax[1].spines["right"].set_visible(False)

  ax[1].xaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)
  ax[1].yaxis.set_tick_params(
      width=gen_linewidth, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax[1].spines[axis].set_linewidth(gen_linewidth)

  for tick in ax[1].xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax[1].yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax[1].set_ylabel("Eval Accuracy (%)",
                   fontsize=font_size, fontweight='bold')

  names = ['a', 'b', 'c', 'd', 'e', 'f']
  ax[1].set_xticks((base_x), names, )
  ax[0].set_xticks((base_x), names, )

  plt.tight_layout()
  plt.savefig('figures/surrogate_grads_24.png', dpi=300)
  plt.close()


if __name__ == '__main__':
  major_ver, minor_ver, _ = version.parse(tb.__version__).release
  assert major_ver >= 2 and minor_ver >= 3, \
      "This notebook requires TensorBoard 2.3 or later."
  print("TensorBoard version: ", tb.__version__)

  mpl.rcParams['font.family'] = 'sans-serif'
  mpl.rcParams['font.sans-serif'] = 'Helvetica'
  mpl.rcParams['font.weight'] = 'bold'
  mpl.rcParams['mathtext.fontset'] = 'custom'
  mpl.rcParams['mathtext.rm'] = 'sans'
  mpl.rcParams['mathtext.it'] = 'sans:bold'
  mpl.rcParams['mathtext.default'] = 'bf'

  plot_surrogate_24()
  plot_surrogate()
  plot_meth_sur()
  plot_comparison('figures/overview.png')
