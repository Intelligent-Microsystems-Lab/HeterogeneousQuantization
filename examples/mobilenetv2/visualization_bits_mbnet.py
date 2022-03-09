# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer


from absl import app
from absl import flags
from absl import logging
from clu import platform

import pickle
import copy

import matplotlib.pyplot as plt

import jax
from jax import random
import tensorflow as tf
import numpy as np

import jax.numpy as jnp

import ml_collections
from ml_collections import config_flags
from flax.core import unfreeze

from collections import OrderedDict

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir)))

from train_utils import (  # noqa: E402
    create_model,
    create_train_state,
    restore_checkpoint,
)
import models  # noqa: E402


# import resource
# low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


mbnet_template = {
    '/stem_conv/weight': (-1, 99, 1),
    '/InvertedResidual_0/QuantConv_0/act': (-1, 76, 0),
    '/InvertedResidual_0/QuantConv_0/weight': (-1, 76, 1),
    '/InvertedResidual_0/QuantConv_1/act': (-1, 99, 0),
    '/InvertedResidual_0/QuantConv_1/weight': (-1, 99, 1),
    '/InvertedResidual_1/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_1/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_1/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_1/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_1/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_1/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_2/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_2/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_2/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_2/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_2/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_2/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_3/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_3/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_3/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_3/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_3/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_3/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_4/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_4/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_4/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_4/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_4/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_4/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_5/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_5/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_5/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_5/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_5/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_5/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_6/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_6/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_6/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_6/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_6/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_6/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_7/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_7/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_7/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_7/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_7/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_7/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_8/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_8/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_8/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_8/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_8/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_8/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_9/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_9/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_9/QuantConv_1/act': (-1, 769, 0),
    '/InvertedResidual_9/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_9/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_9/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_10/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_10/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_10/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_10/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_10/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_10/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_11/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_11/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_11/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_11/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_11/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_11/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_12/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_12/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_12/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_12/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_12/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_12/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_13/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_13/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_13/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_13/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_13/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_13/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_14/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_14/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_14/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_14/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_14/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_14/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_15/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_15/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_15/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_15/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_15/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_15/QuantConv_2/weight': (-1, 99, 1),
    '/InvertedResidual_16/QuantConv_0/act': (-1, 99, 1),
    '/InvertedResidual_16/QuantConv_0/weight': (-1, 99, 1),
    '/InvertedResidual_16/QuantConv_1/act': (-1, 76, 0),
    '/InvertedResidual_16/QuantConv_1/weight': (-1, 76, 1),
    '/InvertedResidual_16/QuantConv_2/act': (-1, 99, 0),
    '/InvertedResidual_16/QuantConv_2/weight': (-1, 99, 1),
    '/head_conv/act': (-1, 99, 1),
    '/head_conv/weight': (-1, 99, 1),
    '/QuantDense_0/act': (-1, 99, 0),
    '/QuantDense_0/weight': (-1, 99, 1),
    '/QuantDense_0/bias': (-1, 99, 1),
}


color_to_label = {
    'blue': '# Bits Weights',
    'green': '# Bits Activation',
    'orange': '# Bits Bias',
}


def flatten_names(x, path):
  if hasattr(x, 'keys'):
    list_x = []
    for y in x.keys():
      list_x += flatten_names(x[y], path + '/' + y)
    return list_x
  return [path]


def fetch_value(x, name_list):
  if len(name_list) == 1:
    return x[name_list[0]]
  else:
    return fetch_value(x[name_list[0]], name_list[1:])


def load_data(config: ml_collections.ConfigDict, workdir: str):
  rng = random.PRNGKey(config.seed)

  model_cls = getattr(models, config.model)
  model = create_model(model_cls=model_cls,
                       num_classes=config.num_classes, config=config)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, lambda x: x)
  state = restore_checkpoint(state, workdir)

  act_names = flatten_names(unfreeze(state.act_size), '')
  flat_acts = {}
  for local_name in act_names:
    flat_acts[local_name] = fetch_value(
        state.act_size, local_name[1:].split('/'))

  weight_names = flatten_names(unfreeze(state.weight_size), '')
  flat_weights = {}
  for local_name in weight_names:
    flat_weights[local_name] = fetch_value(
        state.weight_size, local_name[1:].split('/'))

  param_names = flatten_names(unfreeze(state.params['quant_params']), '')
  flat_params = {}
  for local_name in param_names:
    flat_params[local_name] = fetch_value(
        state.params['quant_params'], local_name[1:].split('/'))

  max_bits = 0
  max_xmax = 0
  max_num = 0

  w_max_bits = 0
  w_max_xmax = 0
  w_max_num = 0

  a_max_bits = 0
  a_max_xmax = 0
  a_max_num = 0

  enet_bits = copy.deepcopy(mbnet_template)
  for i in flat_params:
    if i == '/placeholder':
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '0':
      if enet_bits[i.split('/parametric_d_xmax_')[0] + '/weight'][0] == -1:
        # weights
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(
            state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0] + '/weight'][2]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_weights[i.split(
            '/parametric_d_xmax_0')[0] + '/parametric_d_xmax_0/weight_mb'] \
            / bits_comp
        enet_bits[i.split('/parametric_d_xmax_')[0]
                  + '/weight'] = (bits_comp, xmax, num_params,
                                  enet_bits[i.split('/parametric_d_xmax_')[0]
                                            + '/weight'][1], 'blue')
        if bits_comp > max_bits:
          max_bits = bits_comp
        if xmax > max_xmax:
          max_xmax = xmax
        if num_params > max_num:
          max_num = num_params

        if bits_comp > w_max_bits:
          w_max_bits = bits_comp
        if xmax > w_max_xmax:
          w_max_xmax = xmax
        if num_params > w_max_num:
          w_max_num = num_params
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '1':
      if enet_bits[i.split('/parametric_d_xmax_')[0] + '/act'][0] == -1:
        # act
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(
            state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0] + '/act'][2]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_acts[i.split(
            '/parametric_d_xmax_1')[0] + '/parametric_d_xmax_1/act_mb'] \
            / bits_comp

        enet_bits[i.split('/parametric_d_xmax_')[0]
                  + '/act'] = (bits_comp, xmax, num_params,
                               enet_bits[i.split('/parametric_d_xmax_')[0]
                                         + '/act'][1], 'green')
        if bits_comp > max_bits:
          max_bits = bits_comp
        if xmax > max_xmax:
          max_xmax = xmax
        if num_params > max_num:
          max_num = num_params

        if bits_comp > a_max_bits:
          a_max_bits = bits_comp
        if xmax > a_max_xmax:
          a_max_xmax = xmax
        if num_params > a_max_num:
          a_max_num = num_params
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '2':
      if enet_bits[i.split('/parametric_d_xmax_')[0] + '/bias'][0] == -1:
        # bias
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(
            state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0] + '/bias'][2]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_weights[i.split(
            '/parametric_d_xmax_2')[0] + '/parametric_d_xmax_2/weight_mb'] \
            / bits_comp
        enet_bits[i.split('/parametric_d_xmax_')[0]
                  + '/bias'] = (bits_comp, xmax, num_params,
                                enet_bits[i.split('/parametric_d_xmax_')[0]
                                          + '/bias'][1], 'orange')
        if bits_comp > max_bits:
          max_bits = bits_comp
        if xmax > max_xmax:
          max_xmax = xmax
        if num_params > max_num:
          max_num = num_params

        if bits_comp > w_max_bits:
          w_max_bits = bits_comp
        if xmax > w_max_xmax:
          w_max_xmax = xmax
        if num_params > w_max_num:
          w_max_num = num_params
      continue

  return enet_bits, {'w_max_bits': w_max_bits, 'w_max_xmax': w_max_xmax,
                     'w_max_num': w_max_num, 'a_max_bits': a_max_bits,
                     'a_max_xmax': a_max_xmax, 'a_max_num': a_max_num,
                     'max_bits': max_bits, 'max_xmax': max_xmax,
                     'max_num': max_num}


def plot_bits(config: ml_collections.ConfigDict, workdir: str):

  # try:
  #   with open('/Users/clemens/Desktop/mbnet_bits.pkl', 'rb') as f:
  #     enet_bits = pickle.load(f)
  #   with open('/Users/clemens/Desktop/mbnet_max.pkl', 'rb') as f:
  #     max_data = pickle.load(f)
  # except:
  enet_bits, max_data = load_data(config, workdir)
  with open('/Users/clemens/Desktop/mbnet_bits.pkl', 'wb') as f:
    pickle.dump(enet_bits, f)
  with open('/Users/clemens/Desktop/mbnet_max.pkl', 'wb') as f:
    pickle.dump(max_data, f)

  def plot_fig_num(omit, name):

    font_size = 23
    plt.rc('font', family='Helvetica', weight='bold')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16.5, 6.0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.xaxis.set_tick_params(width=3, length=10, labelsize=font_size)
    ax.yaxis.set_tick_params(width=3, length=10, labelsize=font_size)

    ax.spines['left'].set_position('zero')

    for axis in ['top', 'bottom', 'left', 'right']:
      ax.spines[axis].set_linewidth(5)

    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontweight('bold')

    if '/act' in omit:
      local_max_bits = 7  # max_data['w_max_bits']
      # local_max_xmax = max_data['w_max_xmax']
      local_max_num = 6  # max_data['w_max_num']
      ax.plot([0, 107 + 1], [2 / 7, 2 / 7], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 107 + 1], [4 / 7, 4 / 7], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 107 + 1], [6 / 7, 6 / 7], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)

      ax.plot([0, 107 + 1], [-2 / 6, -2 / 6], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 107 + 1], [-4 / 6, -4 / 6], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 107 + 1], [-6 / 6, -6 / 6], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
    elif '/weight' in omit:
      local_max_bits = 5  # max_data['a_max_bits']
      # local_max_xmax = max_data['a_max_xmax']
      local_max_num = 6  # max_data['a_max_num']

      ax.plot([0, 105 + 1], [1 / 4.1, 1 / 4.1], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 105 + 1], [3 / 4.1, 3 / 4.1], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)

      ax.plot([0, 105 + 1], [-2 / 6, -2 / 6], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 105 + 1], [-4 / 6, -4 / 6], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
      ax.plot([0, 105 + 1], [-6 / 6, -6 / 6], color='grey', alpha=.5,
              linestyle='--', linewidth=3, zorder=0)
    else:
      local_max_bits = 7  # max_data['max_bits']
      # local_max_xmax = max_data['max_xmax']
      local_max_num = max_data['max_num']
      # label_num = '# Activations/Weights'

    pos = 1
    # extra_entry = False
    for k, v in enet_bits.items():
      skip = False
      for iu in omit:
        if iu in k:
          skip = True
      if not skip:

        if v[3] == 76:
          label = 'Depthwise Conv.'
          color = 'g'
        elif ('QuantConv' in k) or ('head_conv' in k):
          label = 'Pointwise Conv.'
          color = 'orange'
        elif ('stem_conv' in k):
          label = '3x3 Convolution'
          color = 'b'
        elif ('QuantDense' in k):
          label = 'Affine'
          color = 'r'
        else:
          raise Exception('Unknown error type: ' + k)

        ax.bar(pos, v[0] / local_max_bits, width=2.0, color=color,
               label=label, edgecolor='black', zorder=10)
        ax.bar(pos, -1 * (np.log10(v[2])) / local_max_num,
               width=2., color=color, label=label, edgecolor='black',
               zorder=10)
        pos += 2

    # print('Done')
    # ax.set_xlabel("#quantization", fontsize=font_size, fontweight='bold')
    # ax.set_ylabel("bits", fontsize=font_size, fontweight='bold')
    ax.text(-5, 0.3, '# Bits', dict(size=23), rotation=90)
    ax.text(-5, -.6, 'log10 #', dict(size=23), rotation=90)
    ax.annotate('', xy=(0, 0), xytext=(pos + 2, 0), arrowprops=dict(
        arrowstyle='<-, head_width=.3, head_length=1.', lw=3), zorder=20)
    ax.annotate('', xy=(0, -1.13), xytext=(0, 1.1), arrowprops=dict(
        arrowstyle='<->, head_width=.3,  head_length=1.', lw=3), zorder=20)
    if '/act' in omit:
      plt.ylim(-1.13, 1.1)
      ax.set_yticks([-6 / 6, -4 / 6, -2 / 6, 0., 2 / 7, 4 / 7, 6 / 7])
      ax.set_yticklabels([6, 4, 2, 0, 2, 4, 6])
      ax.text(33, -1.2, "Weights MobileNetV2", dict(size=23))
    else:
      plt.ylim(-1.13, 1.1)
      ax.set_yticks([-6 / 6, -4 / 6, -2 / 6, 0., 2 / 5, 4 / 5, ])
      ax.set_yticklabels([6, 4, 2, 0, 2, 4])
      ax.text(33, -1.2, "Activations MobileNetV2", dict(size=23))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(.1, .95, .8, 0.11),
               loc="lower left", borderaxespad=0, ncol=4, mode='expand',
               frameon=False,
               prop={'weight': 'bold', 'size': font_size})

    ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()

  def plot_fig_aw(name):

    font_size = 23
    plt.rc('font', family='Helvetica', weight='bold')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16.5, 6.0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.xaxis.set_tick_params(width=3, length=10, labelsize=font_size)
    ax.yaxis.set_tick_params(width=3, length=10, labelsize=font_size)

    ax.spines['left'].set_position('zero')

    for axis in ['top', 'bottom', 'left', 'right']:
      ax.spines[axis].set_linewidth(5)

    for tick in ax.xaxis.get_major_ticks():
      tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
      tick.label1.set_fontweight('bold')

    # if '/act' in omit:
    localw_max_bits = 7  # max_data['w_max_bits']
    # localw_max_xmax = max_data['w_max_xmax']
    # localw_max_num = max_data['w_max_num']
    ax.plot([0, 107 + 1], [2 / 7, 2 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 107 + 1], [4 / 7, 4 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 107 + 1], [6 / 7, 6 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    # elif '/weight' in omit:
    locala_max_bits = 7  # max_data['a_max_bits']
    # locala_max_xmax = max_data['a_max_xmax']
    # locala_max_num = max_data['a_max_num']
    ax.plot([0, 107 + 1], [-2 / 7, -2 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 107 + 1], [-4 / 7, -4 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)

    a_pos = 3
    w_pos = 1
    # extra_entry = False
    for k, v in enet_bits.items():
      if 'bias' not in k:
        if v[3] == 76:
          label = 'Depthwise Conv.'
          color = 'g'
        elif ('QuantConv' in k) or ('head_conv' in k):
          label = 'Pointwise Conv.'
          color = 'orange'
        elif ('stem_conv' in k):
          label = '3x3 Convolution'
          color = 'b'
        elif ('QuantDense' in k):
          label = 'Affine'
          color = 'r'
        else:
          raise Exception('Unknown error type: ' + k)

        if 'act' in k:
          ax.bar(a_pos, -v[0] / locala_max_bits, width=2.0, color=color,
                 label=label, edgecolor='black', zorder=10)
          a_pos += 2
        else:
          ax.bar(w_pos, v[0] / localw_max_bits, width=2.0, color=color,
                 label=label, edgecolor='black', zorder=10)
          w_pos += 2

    # ax.set_xlabel("#quantization", fontsize=font_size, fontweight='bold')
    # ax.set_ylabel("bits", fontsize=font_size, fontweight='bold')
    ax.text(-5, 0.3, 'W # Bits', dict(size=23), rotation=90)
    ax.text(-5, -.6, 'A # Bits', dict(size=23), rotation=90)
    ax.annotate('', xy=(0, 0), xytext=(107 + 2, 0), arrowprops=dict(
        arrowstyle='<-, head_width=.3, head_length=1.', lw=3), zorder=20)
    ax.annotate('', xy=(0, -.8), xytext=(0, 1.05), arrowprops=dict(
        arrowstyle='<->, head_width=.3,  head_length=1.', lw=3), zorder=20)

    plt.ylim(-.8, 1.05)
    ax.set_yticks([-4 / 7, -2 / 7, 0., 2 / 7, 4 / 7, 6 / 7])
    ax.set_yticklabels([4, 2, 0, 2, 4, 6])
    ax.text(37, -.82, "Bits MobileNetV2", dict(size=23))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               bbox_to_anchor=(.1, .95, .8, 0.11),
               loc="lower left", borderaxespad=0, ncol=4, mode='expand',
               frameon=False,
               prop={'weight': 'bold', 'size': font_size})

    ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    plt.tight_layout()
    plt.savefig(name, dpi=300)
    plt.close()

  # plot_fig_num(['/bias'], 'mobilenetv2/figures/bitwidths_all.png')
  plot_fig_num(['/act', '/bias'], 'mobilenetv2/figures/bitwidths_w.png')
  plot_fig_num(['/weight', '/bias'], 'mobilenetv2/figures/bitwidths_a.png')
  plot_fig_aw('mobilenetv2/figures/bitwidths_aw.png')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'proc_index: {jax.process_index()}, '
                                       f'proc_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  plot_bits(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
