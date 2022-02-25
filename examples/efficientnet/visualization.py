import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import tensorboard as tb
from packaging import version
import grpc

import numpy as np
import itertools

# Competitor Performance.
competitors = {
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

    'pact_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125]) * 100,
        'size_mb': np.array([4, 5, 6
                             ]) * 3472041 / 8_000_000 + 0.13644800 / 2,
        'name': 'PACT MbNetV2*',  # 'PACT MobileNetV2*',
        'alpha': .25,
    },

    'dsq_resnet18': {
        # https://arxiv.org/abs/1908.05033
        'eval_err': np.array([1 - 0.6517, 1 - 0.6866, 1 - 0.6956]) * 100,
        'size_mb': np.array([2, 3, 4
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'DSQ ResNet18*',
        'alpha': .25,
    },

    'lsq_resnet18': {
        # https://arxiv.org/abs/1902.08153
        'eval_err': np.array([1 - 0.676, 1 - 0.702, 1 - 0.711,
                              1 - 0.711]) * 100,
        'size_mb': np.array([2, 3, 4, 8
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'LSQ ResNet18*',
        'alpha': .25,
    },

    # 'lsqp_resnet18': {
    #     # https://arxiv.org/abs/2004.09576
    #     # they claim to be a natural extension... so also first and last?
    #     'eval_err': np.array([1 - 0.668, 1 - 0.694, 1 - 0.708]) * 100,
    #     'size_mb': np.array([2, 3, 4]) * 11679912 / 8_000_000 + 0.03840000 / 2,
    #     'name': 'LSQ+ ResNet18*',
    #     'alpha': .25,
    # },

    'lsqp_enet0': {
        # this is efficient-b0 not lite (!)
        # https://arxiv.org/abs/2004.09576
        'eval_err': np.array([1 - 0.491, 1 - 0.699, 1 - 0.738]) * 100,
        # number might be incorrect
        'size_mb': np.array([2, 3, 4]) * 5246532 / 8_000_000 + 0.16806400 / 2,
        'name': 'LSQ+ ENet-B0*',
        'alpha': .25,
    },

    'ewgs_resnet18': {
        # https://arxiv.org/abs/2104.00903
        'eval_err': np.array([1 - 0.553, 1 - 0.67, 1 - 0.697,
                              1 - 0.706]) * 100,
        'size_mb': np.array([1, 2, 3, 4
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'EWGS ResNet18*',
        'alpha': .25,
    },


    'psgd_resnet18': {
        # https://arxiv.org/pdf/2005.11035.pdf
        'eval_err': np.array([1 - 0.6345, 1 - 0.6951, 1 - 0.7013, ]) * 100,
        'size_mb': np.array([4, 6, 8]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'PSGD ResNet18',
        'alpha': .25,
    },

    # 'ewgs_resnet34': {
    #     # https://arxiv.org/abs/2104.00903
    #     'eval_err': np.array([1 - 0.615, 1 - 0.714, 1 - 0.733,
    #                           1 - 0.739]) * 100,
    #     'size_mb': np.array([1, 2, 3, 4
    #                          ]) * 21780648 / 8_000_000 + 0.06809600 / 2,
    #     'name': 'EWGS ResNet34*',
    #     'alpha': .25,
    # },

    'qil_resnet18': {
        # "We did not quantize the first and the last layers as was done in..."
        # https://arxiv.org/abs/1808.05779
        'eval_err': np.array([1 - 0.657, 1 - 0.692, 1 - 0.701, 1 - 0.704,
                              ]) * 100,
        'size_mb': np.array([2, 3, 4, 5
                             ]) * 11679912 / 8_000_000 + 0.03840000 / 2,
        'name': 'QIL ResNet18*',
        'alpha': .25,
    },

    # 'qil_resnet34': {
    #     # "We did not quantize the first and the last layers as was done in..."
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
    #     'size_mb': np.array([4, 5, 8]) * 4211113 / 8_000_000 + 0.08755200 / 2,
    #     'name': 'PROFIT MobileNetV1',
    #     'alpha': .25,
    # },



    # 'profit_mobilev3': {
    #     # https://arxiv.org/pdf/2008.04693.pdf
    #     'eval_err': np.array([1 - 0.6139, 1 - 0.6884, 1 - 0.7125, ]) * 100,
    #     'size_mb': np.array([4, 5, 8]) * 5459913 / 8_000_000 + 0.09760000 / 2,
    #     'name': 'PROFIT MobileNetV3',
    #     'alpha': .25,
    # },



}

competitors_bigger_act = {
    # profit doesnt quantize s. We did not apply quantization only for the
    # input image of the first convolution layer and the activation of
    # the squeeze-excitation module.
    'profit_mobilev2': {
        # https://arxiv.org/pdf/2008.04693.pdf
        'eval_err': np.array([1 - .71564, 1 - .72192, 1 - .72352]) * 100,
        'size_mb': np.array([4, 5, 8]) * 3506153 / 8_000_000 + 0.13644800 / 2,
        'name': 'PROFIT MbNetV2',
        'alpha': .25,
    },
    'profit_mnas': {
        # https://arxiv.org/pdf/2008.04693.pdf
        'eval_err': np.array([1 - .72244, 1 - .73378, 1 - .73742]) * 100,
        'size_mb': np.array([4, 5, 8]) * 3853550 / 8_000_000 + 0.13395200 / 2,
        'name': 'PROFIT MNasNet-A1',
        'alpha': .25,
    },

    'hawqv2_squeeze': {
        # activation bits 8 (uniform)
        # https://arxiv.org/abs/1911.03852
        'eval_err': np.array([1 - 0.6838]) * 100,
        'size_mb': np.array([1.07]),
        'name': 'HAWQ-V2 SqueezeNext',
        'alpha': .25,
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
        'eval_err': np.array([0.2992]) * 100,
        'size_mb': np.array([5.4]),
        'name': 'Mixed ResNet18',
        'alpha': .25,
    },

    'mixed_mobilev2': {
        # activation budget 570 KB against
        # https://arxiv.org/abs/1905.11452
        'eval_err': np.array([0.3026]) * 100,
        'size_mb': np.array([1.55]),
        'name': 'Mixed MbNetV2',
        'alpha': .25,
    },

    'haq_mobilev2': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': np.array([1 - 0.6675, 1 - 0.7090, 1 - 0.7147]) * 100,
        'size_mb': np.array([.95, 1.38, 1.79]),
        'name': 'HAQ MbNetV2',
        'alpha': .25,
    },

    'haq_resnet50': {
        # https://arxiv.org/pdf/1811.08886.pdf
        'eval_err': np.array([1 - 0.7063, 1 - 0.7530, 1 - 0.7614]) * 100,
        'size_mb': np.array([6.30, 9.22, 12.14]),
        'name': 'HAQ ResNet50',
        'alpha': .25,
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

  mu = np.nanmean(data, axis=0)
  sigma = np.nanstd(data, axis=0)

  font_size = 23
  plt.rc('font', family='Helvetica', weight='bold')
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
  handles.append(mpatches.Patch(
      facecolor='m', label='Wall-clock Overhead', edgecolor='black', linewidth=3.))

  plt.legend(
      handles=handles,
      bbox_to_anchor=(0., 1.02, 1.0, .1),
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

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  ax2.spines["top"].set_visible(False)

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
  plt.rc('font', family='Helvetica', weight='bold')
  fig, ax = plt.subplots(figsize=(16.5, 8.5))

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
  ax.plot(np.array([4609992, 5363016, 6033976, 8120424, 12894200]
                   ) * 8 / 8_000_000 + np.array([0.16806400, 0.21465600,
                                                 0.23238400,
                                                 0.30668800, 0.44947200]) / 2,
          np.array([0.2452799677848816, 0.2303873896598816, 0.2221272587776184,
                    0.2054443359375, 0.1907958984375]) * 100, marker='x',
          label='ENet0-4 INT8', ms=20, markeredgewidth=5,
          linewidth=5)

  ax.plot(0.5762490000 * np.array([3, 4, 5, 6, 7, 8]) + 0.16806400 / 2,
          np.array([0.36395263671875, 0.2801310420036316, 0.2576497197151184,
                    0.2494099736213684, 0.2458088994026184,
                    0.2452799677848816]) * 100, marker='x',
          label='ENet0 3-8 Bits', ms=20, markeredgewidth=5,
          linewidth=5)

  xv = np.array([1.1537720947265625, 1.1611640625, 1.294680054, 1.700428101,
                 1.832848022,
                2.013916138, 2.142487061, 2.306124023]) + 0.16806400 / 2
  yv = 100 - np.array([0.37939453125, 0.4444986879825592, 0.6106770635,
                       0.679361999, 0.6814778447,
                      0.6925455928, 0.7124023438, 0.7234700322]) * 100
  ax.plot(xv, yv, marker='x', label="M ENet STE",
          ms=20, markeredgewidth=5, linewidth=5, color='red')

  xv = np.array([1.153934082, 1.294906128, 1.417925903, 1.719146118,
                1.874620117, 1.982543091, 2.144371094, 2.290263916]
                ) + 0.16806400 / 2
  yv = 100 - np.array([0.46598, 0.607747376, 0.654622376, 0.6850585938,
                      0.693033874, 0.7049153447, 0.7202148438, 0.7312825322]
                      ) * 100
  ax.plot(xv, yv, marker='x',
          label="M ENet Surrogate",
          ms=20, markeredgewidth=5, linewidth=5, color='green')

  xv = np.array([2.919978271, 3.278890381, 3.50227417, 3.538114014,
                 4.284223145,
                4.539738281, 4.855490234, 5.462999512, 5.81514502]
                ) + 0.03840000 / 2
  yv = 100 - np.array([0.5455729365, 0.5716145635, 0.6126301885, 0.6258137822,
                      0.6560872197, 0.6430664063, 0.6513671875, 0.6647135615,
                      0.6805012822]) * 100
  ax.plot(xv, yv, marker='x', label="M ResNet18 STE",
          ms=20, markeredgewidth=5, linewidth=5, color='violet')

  ax.set_xscale('log')
  plt.xticks([1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
             '1.5', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
  ax.set_xlabel("Network Size (MB)", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("Eval Error (%)", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(-.05, 1.02, 1.05, 0.2),
      loc="lower left",
      ncol=4,
      mode="expand",
      borderaxespad=0,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(name, dpi=300)
  plt.close()


def plot_comparison2(name):
  font_size = 23
  plt.rc('font', family='Helvetica', weight='bold')
  fig, ax = plt.subplots(figsize=(16.5, 8.5))

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
  for competitor_name, competitor_data in competitors_bigger_act.items():
    ax.plot(competitor_data['size_mb'], competitor_data['eval_err'],
            label=competitor_data['name'],
            marker='.', ms=20, markeredgewidth=5, linewidth=5,
            alpha=competitor_data['alpha'])

  # Our own.
  ax.plot(np.array([4609992, 5363016, 6033976, 8120424, 12894200]
                   ) * 8 / 8_000_000 + np.array([0.16806400, 0.21465600,
                                                 0.23238400,
                                                 0.30668800, 0.44947200]) / 2,
          np.array([0.2452799677848816, 0.2303873896598816, 0.2221272587776184,
                    0.2054443359375, 0.1907958984375]) * 100, marker='x',
          label='ENet0-4 INT8', ms=20, markeredgewidth=5,
          linewidth=5)

  ax.plot(0.5762490000 * np.array([3, 4, 5, 6, 7, 8]) + 0.16806400 / 2,
          np.array([0.36395263671875, 0.2801310420036316, 0.2576497197151184,
                    0.2494099736213684, 0.2458088994026184,
                    0.2452799677848816]) * 100, marker='x',
          label='ENet0 3-8 Bits', ms=20, markeredgewidth=5,
          linewidth=5)

  xv = np.array([1153.9700927734375, 1268.692017, 1437.639038, 1574.033081,
                1702.221069, 1865.83606, 1986.667114, 2126.347412,
                2267.265381]) / 1000 + 0.16806400 / 2
  yv = 100 - np.array([0.6048176884651184, 0.6778971553, 0.6868489385,
                      0.6979166865, 0.7075195313, 0.706705749, 0.719075501,
                      0.7220051885, 0.7236328125]) * 100
  ax.plot(xv, yv, marker='x', label="M ENet0 INT4",
          ms=20, markeredgewidth=5, linewidth=5, color='red')

  xv = np.array([1154.7200927734375, 1271.812012, 1440.932129, 1577.504028,
                1664.144897, 1828.402222, 1880.366089, 2146.508057,
                2294.529053]) / 1000 + 0.16806400 / 2
  yv = 100 - np.array([0.6427409052848816, 0.693033874, 0.7003580928,
                      0.70703125, 0.7194010615, 0.7309570313, 0.7298176885,
                      0.7355143428, 0.7364909053]) * 100
  ax.plot(xv, yv, marker='x', label="M ENet0 INT8",
          ms=20, markeredgewidth=5, linewidth=5, color='green')

  ax.set_xscale('log')
  plt.xticks([1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10], [
             '1.5', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
  ax.set_xlabel("Network Size (MB)", fontsize=font_size, fontweight='bold')
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
  data = np.genfromtxt('figures/surrogate_mixed.csv',
                       delimiter=',', names=True)
  names = data.dtype.names
  names = ['W: ' + x.split('_')[0] + ' A: ' + x.split('_')[1] for x in names]
  data = np.genfromtxt('figures/surrogate_mixed.csv',
                       delimiter=',', skip_header=1) * 100
  y = data.flatten(order='F')
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  mu = np.nanmean(data, axis=0)
  sigma = np.nanstd(data, axis=0)

  font_size = 23
  plt.rc('font', family='Helvetica', weight='bold')
  fig, ax = plt.subplots(figsize=(16.5, 11.5))

  ax.scatter(x / 4, y, marker='x', linewidths=5,
             s=180, color='blue', label='Observations')

  ax.scatter(base_x / 4, mu, marker='_', linewidths=5,
             s=840, color='red', label='Mean')

  ax.scatter(base_x / 4, mu + sigma, marker='_', linewidths=5,
             s=840, color='green', label='Std. Dev.')

  ax.scatter(base_x / 4, mu - sigma, marker='_',
             linewidths=5, s=840, color='green')

  ax.axhline(y=65.741000, color='orange',
             linestyle='--', label='STE', linewidth=5)

  ax.axhline(y=66.37000, color='purple',
             linestyle='--', label='EWGS', linewidth=5)

  plt.xticks(base_x / 4, names, rotation=45, horizontalalignment='right')

  plt.legend(
      bbox_to_anchor=(0., 1.02, 1.0, .05),
      loc="upper center",
      ncol=5,
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

  ax.set_ylabel("Eval Accuracy (%)", fontsize=font_size, fontweight='bold')
  plt.tight_layout()
  plt.savefig('figures/surrogate_grads_mixed.png', dpi=300)
  plt.close()


def flip(items, ncol):
  return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def plot_surrogate_24(num):
  data = np.genfromtxt('figures/surrogate_plain' + str(num) + '.csv',
                       delimiter=',', names=True)
  names = data.dtype.names
  data = np.genfromtxt('figures/surrogate_plain' + str(num) + '.csv',
                       delimiter=',', skip_header=1) * 100
  y = data.flatten(order='F')
  x = np.repeat(np.arange(len(names)) + 1, 20)
  base_x = np.arange(len(names)) + 1

  mu = np.nanmean(data, axis=0)
  sigma = np.nanstd(data, axis=0)

  font_size = 23
  plt.rc('font', family='Helvetica', weight='bold')

  fig, ax = plt.subplots(figsize=(8.0, 12.0))

  ax.scatter(base_x / 2, mu, marker='_', linewidths=5,
             s=840, color='red', label='Mean', zorder=10)

  ax.scatter(x / 2, y, marker='x', linewidths=5,
             s=180, color='blue', label='Observations', zorder=0)

  ax.scatter(base_x / 2, mu + sigma, marker='_', linewidths=5,
             s=840, color='green', label='Std. Dev.', zorder=10)

  ax.scatter(base_x / 2, mu - sigma, marker='_',
             linewidths=5, s=840, color='green', zorder=10)

  # ax2 = ax.twinx()

  # data = np.genfromtxt('figures/surrogate_plain2.csv',
  #                      delimiter=',', names=True)
  # names = data.dtype.names
  # data = np.genfromtxt('figures/surrogate_plain2.csv',
  #                      delimiter=',', skip_header=1) * 100
  # y = data.flatten(order='F')
  # x_old = x
  # x = np.repeat(np.arange(x[-1], len(names) + x[-1]) + 1, 20)
  # base_x = np.arange(x_old[-1], len(names) + x_old[-1]) + 1

  # mu = np.nanmean(data, axis=0)
  # sigma = np.nanstd(data, axis=0)

  # ax2.scatter(x / 2, y, marker='x', linewidths=5,
  #             s=180, color='blue', label='Observations 2 Bit', alpha=.35)

  # ax2.scatter(base_x / 2, mu, marker='_', linewidths=5,
  #             s=840, color='red', label='Mean 2 Bit', alpha=.35)

  # ax2.scatter(base_x / 2, mu + sigma, marker='_', linewidths=5,
  #             s=840, color='green', label='Std. Dev. 2 Bit', alpha=.35)

  # ax2.scatter(base_x / 2, mu - sigma, marker='_',
  #             linewidths=5, s=840, color='green', alpha=.35)

  # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

  # handles, labels = ax.get_legend_handles_labels()
  # handles2, labels2 = ax2.get_legend_handles_labels()
  # handles += handles2
  # labels += labels2

  plt.legend(
      # handles=list(flip(handles, 3)),
      bbox_to_anchor=(0., 1.02, 1.0, .1),
      loc="upper center",
      ncol=2,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size},
  )

  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  # ax2.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  # ax2.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  # for axis in ['top', 'bottom', 'left', 'right']:
  #   ax2.spines[axis].set_linewidth(5)

  # for tick in ax2.xaxis.get_major_ticks():
  #   tick.label1.set_fontweight('bold')
  # for tick in ax2.yaxis.get_major_ticks():
  #   tick.label1.set_fontweight('bold')

  ax.set_ylabel("Eval Accuracy (%)",
                fontsize=font_size, fontweight='bold')
  # ax2.set_ylabel("2 Bit Eval Accuracy (%)",
  #                fontsize=font_size, fontweight='bold')

  names = [x.replace('W_', 'W: ').replace('_A_', ' A: ') for x in names]
  plt.xticks((base_x) / 2, names, rotation=45, horizontalalignment='right')
  #plt.xticks((np.arange(0, len(names) + x_old[-1]) + 1) / 2, names)

  fig.autofmt_xdate(rotation=45)
  plt.tight_layout()
  plt.savefig('figures/surrogate_grads_' + str(num) + '.png', dpi=300)
  plt.close()


if __name__ == '__main__':
  major_ver, minor_ver, _ = version.parse(tb.__version__).release
  assert major_ver >= 2 and minor_ver >= 3, \
      "This notebook requires TensorBoard 2.3 or later."
  print("TensorBoard version: ", tb.__version__)

  plot_surrogate_24(2)
  plot_surrogate_24(4)
  plot_surrogate()
  plot_surrogate_mix()
  plot_comparison('figures/overview.png')
  plot_comparison2('figures/overview_act.png')
