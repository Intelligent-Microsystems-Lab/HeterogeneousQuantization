# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer


from absl import app
from absl import flags
from absl import logging
from clu import platform


import copy


import matplotlib.pyplot as plt


import jax
from jax import random
import tensorflow as tf

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


enet_template = {
    '/stem_conv/weight': (-1, 1, 1),
    '/MBConvBlock_0/QuantConv_0/weight': (-1, 2, 1),
    '/MBConvBlock_0/QuantConv_0/act': (-1, 3, 0),
    '/MBConvBlock_0/depthwise_conv2d/weight': (-1, 4, 1),
    '/MBConvBlock_0/depthwise_conv2d/act': (-1, 5, 1),
    '/MBConvBlock_1/QuantConv_0/weight': (-1, 6, 1),
    '/MBConvBlock_1/QuantConv_0/act': (-1, 7, 1),
    '/MBConvBlock_1/QuantConv_1/weight': (-1, 8, 1),
    '/MBConvBlock_1/QuantConv_1/act': (-1, 9, 0),
    '/MBConvBlock_1/depthwise_conv2d/weight': (-1, 10, 1),
    '/MBConvBlock_1/depthwise_conv2d/act': (-1, 11, 1),
    '/MBConvBlock_10/QuantConv_0/weight': (-1, 12, 1),
    '/MBConvBlock_10/QuantConv_0/act': (-1, 13, 1),
    '/MBConvBlock_10/QuantConv_1/weight': (-1, 14, 1),
    '/MBConvBlock_10/QuantConv_1/act': (-1, 15, 0),
    '/MBConvBlock_10/depthwise_conv2d/weight': (-1, 16, 1),
    '/MBConvBlock_10/depthwise_conv2d/act': (-1, 17, 1),
    '/MBConvBlock_11/QuantConv_0/weight': (-1, 18, 1),
    '/MBConvBlock_11/QuantConv_0/act': (-1, 19, 1),
    '/MBConvBlock_11/QuantConv_1/weight': (-1, 20, 1),
    '/MBConvBlock_11/QuantConv_1/act': (-1, 21, 0),
    '/MBConvBlock_11/depthwise_conv2d/weight': (-1, 22, 1),
    '/MBConvBlock_11/depthwise_conv2d/act': (-1, 23, 1),
    '/MBConvBlock_12/QuantConv_0/weight': (-1, 24, 1),
    '/MBConvBlock_12/QuantConv_0/act': (-1, 25, 1),
    '/MBConvBlock_12/QuantConv_1/weight': (-1, 26, 1),
    '/MBConvBlock_12/QuantConv_1/act': (-1, 27, 0),
    '/MBConvBlock_12/depthwise_conv2d/weight': (-1, 28, 1),
    '/MBConvBlock_12/depthwise_conv2d/act': (-1, 29, 1),
    '/MBConvBlock_13/QuantConv_0/weight': (-1, 30, 1),
    '/MBConvBlock_13/QuantConv_0/act': (-1, 31, 1),
    '/MBConvBlock_13/QuantConv_1/weight': (-1, 32, 1),
    '/MBConvBlock_13/QuantConv_1/act': (-1, 33, 0),
    '/MBConvBlock_13/depthwise_conv2d/weight': (-1, 34, 1),
    '/MBConvBlock_13/depthwise_conv2d/act': (-1, 35, 1),
    '/MBConvBlock_14/QuantConv_0/weight': (-1, 36, 1),
    '/MBConvBlock_14/QuantConv_0/act': (-1, 37, 1),
    '/MBConvBlock_14/QuantConv_1/weight': (-1, 38, 1),
    '/MBConvBlock_14/QuantConv_1/act': (-1, 39, 0),
    '/MBConvBlock_14/depthwise_conv2d/weight': (-1, 40, 1),
    '/MBConvBlock_14/depthwise_conv2d/act': (-1, 41, 1),
    '/MBConvBlock_15/QuantConv_0/weight': (-1, 42, 1),
    '/MBConvBlock_15/QuantConv_0/act': (-1, 43, 1),
    '/MBConvBlock_15/QuantConv_1/weight': (-1, 44, 1),
    '/MBConvBlock_15/QuantConv_1/act': (-1, 45, 0),
    '/MBConvBlock_15/depthwise_conv2d/weight': (-1, 46, 1),
    '/MBConvBlock_15/depthwise_conv2d/act': (-1, 47, 1),
    '/MBConvBlock_2/QuantConv_0/weight': (-1, 48, 1),
    '/MBConvBlock_2/QuantConv_0/act': (-1, 49, 1),
    '/MBConvBlock_2/QuantConv_1/weight': (-1, 50, 1),
    '/MBConvBlock_2/QuantConv_1/act': (-1, 51, 0),
    '/MBConvBlock_2/depthwise_conv2d/weight': (-1, 52, 1),
    '/MBConvBlock_2/depthwise_conv2d/act': (-1, 53, 1),
    '/MBConvBlock_3/QuantConv_0/weight': (-1, 54, 1),
    '/MBConvBlock_3/QuantConv_0/act': (-1, 55, 1),
    '/MBConvBlock_3/QuantConv_1/weight': (-1, 56, 1),
    '/MBConvBlock_3/QuantConv_1/act': (-1, 57, 0),
    '/MBConvBlock_3/depthwise_conv2d/weight': (-1, 58, 1),
    '/MBConvBlock_3/depthwise_conv2d/act': (-1, 59, 1),
    '/MBConvBlock_4/QuantConv_0/weight': (-1, 60, 1),
    '/MBConvBlock_4/QuantConv_0/act': (-1, 61, 1),
    '/MBConvBlock_4/QuantConv_1/weight': (-1, 62, 1),
    '/MBConvBlock_4/QuantConv_1/act': (-1, 63, 0),
    '/MBConvBlock_4/depthwise_conv2d/weight': (-1, 64, 1),
    '/MBConvBlock_4/depthwise_conv2d/act': (-1, 65, 1),
    '/MBConvBlock_5/QuantConv_0/weight': (-1, 66, 1),
    '/MBConvBlock_5/QuantConv_0/act': (-1, 67, 1),
    '/MBConvBlock_5/QuantConv_1/weight': (-1, 68, 1),
    '/MBConvBlock_5/QuantConv_1/act': (-1, 69, 0),
    '/MBConvBlock_5/depthwise_conv2d/weight': (-1, 70, 1),
    '/MBConvBlock_5/depthwise_conv2d/act': (-1, 71, 1),
    '/MBConvBlock_6/QuantConv_0/weight': (-1, 72, 1),
    '/MBConvBlock_6/QuantConv_0/act': (-1, 73, 1),
    '/MBConvBlock_6/QuantConv_1/weight': (-1, 74, 1),
    '/MBConvBlock_6/QuantConv_1/act': (-1, 75, 0),
    '/MBConvBlock_6/depthwise_conv2d/weight': (-1, 76, 1),
    '/MBConvBlock_6/depthwise_conv2d/act': (-1, 77, 1),
    '/MBConvBlock_7/QuantConv_0/weight': (-1, 78, 1),
    '/MBConvBlock_7/QuantConv_0/act': (-1, 79, 1),
    '/MBConvBlock_7/QuantConv_1/weight': (-1, 80, 1),
    '/MBConvBlock_7/QuantConv_1/act': (-1, 81, 0),
    '/MBConvBlock_7/depthwise_conv2d/weight': (-1, 82, 1),
    '/MBConvBlock_7/depthwise_conv2d/act': (-1, 83, 1),
    '/MBConvBlock_8/QuantConv_0/weight': (-1, 84, 1),
    '/MBConvBlock_8/QuantConv_0/act': (-1, 85, 1),
    '/MBConvBlock_8/QuantConv_1/weight': (-1, 86, 1),
    '/MBConvBlock_8/QuantConv_1/act': (-1, 87, 0),
    '/MBConvBlock_8/depthwise_conv2d/weight': (-1, 88, 1),
    '/MBConvBlock_8/depthwise_conv2d/act': (-1, 89, 1),
    '/MBConvBlock_9/QuantConv_0/weight': (-1, 90, 1),
    '/MBConvBlock_9/QuantConv_0/act': (-1, 91, 1),
    '/MBConvBlock_9/QuantConv_1/weight': (-1, 92, 1),
    '/MBConvBlock_9/QuantConv_1/act': (-1, 93, 0),
    '/MBConvBlock_9/depthwise_conv2d/weight': (-1, 94, 1),
    '/MBConvBlock_9/depthwise_conv2d/act': (-1, 95, 1),
    '/head_conv/weight': (-1, 96, 1),
    '/head_conv/act': (-1, 97, 1),
    # '/act': (-1, 98, 0),
    '/QuantDense_0/weight': (-1, 99, 1),
    '/QuantDense_0/act': (-1, 100, 0),
    '/QuantDense_0/bias': (-1, 101, 1),
}


color_to_label = {
    'red': 'Average',
    'blue': 'Weights',
    'green': 'Activation',
    'orange': 'Bias',
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


def plot_bits(config: ml_collections.ConfigDict, workdir: str):
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

  enet_bits = copy.deepcopy(enet_template)
  for i in flat_params:
    if i == '/placeholder':
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '0':
      if enet_bits[i.split('/parametric_d_xmax_')[0] + '/weight'][0] == -1:
        # weights
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0] + '/weight'][2]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_weights[ i.split('/parametric_d_xmax_0')[0] + '/parametric_d_xmax_0/weight_mb'] / bits_comp
        enet_bits[i.split('/parametric_d_xmax_')[0] + '/weight'] = (bits_comp, xmax, num_params,
            enet_bits[i.split('/parametric_d_xmax_')[0] + '/weight'][1], 'blue')
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '1':
      if enet_bits[i.split('/parametric_d_xmax_')[0] + '/act'][0] == -1:
        # act
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0] + '/act'][2]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params =  flat_acts[ i.split('/parametric_d_xmax_1')[0] + '/parametric_d_xmax_0/act_mb'] /  bits_comp

        enet_bits[i.split('/parametric_d_xmax_')[0] + '/act'] = (bits_comp, xmax, num_params,
            enet_bits[i.split('/parametric_d_xmax_')[0] + '/act'][1], 'green')
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '2':
      if enet_bits[i.split('/parametric_d_xmax_')[0] + '/bias'][0] == -1:
        # bias
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0] + '/bias'][2]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_weights[ i.split('/parametric_d_xmax_2')[0] + '/parametric_d_xmax_0/weight_mb'] /  bits_comp
        enet_bits[i.split('/parametric_d_xmax_')[0] + '/bias'] = (bits_comp, xmax, num_params,
            enet_bits[i.split('/parametric_d_xmax_')[0] + '/bias'][1], 'orange')
      continue
    import pdb; pdb.set_trace()


  # def plot_fig(omit, name):
  font_size = 22

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 5.0))
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

  pos = 1
  for k, v in enet_bits.items():
    ax.bar(pos, v[0], width=1.0, color=v[-1], label=color_to_label[v[-1]])
    ax.bar(pos, -v[1], width=.5, color=v[-1], label=color_to_label[v[-1]])
    ax.bar(pos+.5, -v[2], width=.5, color=v[-1], label=color_to_label[v[-1]])
    pos += 1

  ax.set_xlabel("#quantization", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("bits", fontsize=font_size, fontweight='bold')

  handles, labels = plt.gca().get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  plt.legend(by_label.values(), by_label.keys(),
             bbox_to_anchor=(0, 1.02, 1, 0.2),
             loc="lower left", borderaxespad=0, ncol=5, mode='expand',
             frameon=False,
             prop={'weight': 'bold', 'size': font_size})

  plt.tight_layout()
  plt.savefig('efficientnet/figures/bitwidths_all.png')
  plt.close()

  # plot_fig([], 'efficientnet/figures/bitwidths_all.png')
  # plot_fig(['/act'], 'efficientnet/figures/bitwidths_w.png')
  # plot_fig(['/weight', '/bias'], 'efficientnet/figures/bitwidths_a.png')


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
