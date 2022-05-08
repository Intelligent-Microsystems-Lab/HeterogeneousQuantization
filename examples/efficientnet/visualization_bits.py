# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

from absl import app

import pickle
import copy

import matplotlib.pyplot as plt

import jax
from jax import random
import jax.numpy as jnp

import ml_collections
from flax.core import unfreeze

from collections import OrderedDict

import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../mobilenetv2')

from train_utils import (  # noqa: E402
    create_model,
    create_train_state,
    restore_checkpoint,
)
import models as enet_models  # noqa: E402
import configs.vis_dummy_mb as mb_conf  # noqa: E402
import configs.vis_dummy as enet_conf  # noqa: E402
import mobilenetv2.models as mbnet_model  # noqa: E402

enet_template = {
    '/stem_conv/weight': (-1, 1, 1),
    '/MBConvBlock_0/QuantConv_0/act': (-1, 3, 0),
    '/MBConvBlock_0/QuantConv_0/weight': (-1, 2, 1),
    '/MBConvBlock_0/depthwise_conv2d/act': (-1, 5, 1),
    '/MBConvBlock_0/depthwise_conv2d/weight': (-1, 4, 1),
    '/MBConvBlock_1/QuantConv_0/act': (-1, 7, 1),
    '/MBConvBlock_1/QuantConv_0/weight': (-1, 6, 1),
    '/MBConvBlock_1/QuantConv_1/act': (-1, 9, 0),
    '/MBConvBlock_1/QuantConv_1/weight': (-1, 8, 1),
    '/MBConvBlock_1/depthwise_conv2d/act': (-1, 11, 1),
    '/MBConvBlock_1/depthwise_conv2d/weight': (-1, 10, 1),
    '/MBConvBlock_2/QuantConv_0/act': (-1, 49, 1),
    '/MBConvBlock_2/QuantConv_0/weight': (-1, 48, 1),
    '/MBConvBlock_2/QuantConv_1/act': (-1, 51, 0),
    '/MBConvBlock_2/QuantConv_1/weight': (-1, 50, 1),
    '/MBConvBlock_2/depthwise_conv2d/act': (-1, 53, 1),
    '/MBConvBlock_2/depthwise_conv2d/weight': (-1, 52, 1),
    '/MBConvBlock_3/QuantConv_0/act': (-1, 55, 1),
    '/MBConvBlock_3/QuantConv_0/weight': (-1, 54, 1),
    '/MBConvBlock_3/QuantConv_1/act': (-1, 57, 0),
    '/MBConvBlock_3/QuantConv_1/weight': (-1, 56, 1),
    '/MBConvBlock_3/depthwise_conv2d/act': (-1, 59, 1),
    '/MBConvBlock_3/depthwise_conv2d/weight': (-1, 58, 1),
    '/MBConvBlock_4/QuantConv_0/act': (-1, 61, 1),
    '/MBConvBlock_4/QuantConv_0/weight': (-1, 60, 1),
    '/MBConvBlock_4/QuantConv_1/act': (-1, 63, 0),
    '/MBConvBlock_4/QuantConv_1/weight': (-1, 62, 1),
    '/MBConvBlock_4/depthwise_conv2d/act': (-1, 65, 1),
    '/MBConvBlock_4/depthwise_conv2d/weight': (-1, 64, 1),
    '/MBConvBlock_5/QuantConv_0/act': (-1, 67, 1),
    '/MBConvBlock_5/QuantConv_0/weight': (-1, 66, 1),
    '/MBConvBlock_5/QuantConv_1/act': (-1, 69, 0),
    '/MBConvBlock_5/QuantConv_1/weight': (-1, 68, 1),
    '/MBConvBlock_5/depthwise_conv2d/act': (-1, 71, 1),
    '/MBConvBlock_5/depthwise_conv2d/weight': (-1, 70, 1),
    '/MBConvBlock_6/QuantConv_0/act': (-1, 73, 1),
    '/MBConvBlock_6/QuantConv_0/weight': (-1, 72, 1),
    '/MBConvBlock_6/QuantConv_1/act': (-1, 75, 0),
    '/MBConvBlock_6/QuantConv_1/weight': (-1, 74, 1),
    '/MBConvBlock_6/depthwise_conv2d/act': (-1, 77, 1),
    '/MBConvBlock_6/depthwise_conv2d/weight': (-1, 76, 1),
    '/MBConvBlock_7/QuantConv_0/act': (-1, 79, 1),
    '/MBConvBlock_7/QuantConv_0/weight': (-1, 78, 1),
    '/MBConvBlock_7/QuantConv_1/act': (-1, 81, 0),
    '/MBConvBlock_7/QuantConv_1/weight': (-1, 80, 1),
    '/MBConvBlock_7/depthwise_conv2d/act': (-1, 83, 1),
    '/MBConvBlock_7/depthwise_conv2d/weight': (-1, 82, 1),
    '/MBConvBlock_8/QuantConv_0/act': (-1, 85, 1),
    '/MBConvBlock_8/QuantConv_0/weight': (-1, 84, 1),
    '/MBConvBlock_8/QuantConv_1/act': (-1, 87, 0),
    '/MBConvBlock_8/QuantConv_1/weight': (-1, 86, 1),
    '/MBConvBlock_8/depthwise_conv2d/act': (-1, 89, 1),
    '/MBConvBlock_8/depthwise_conv2d/weight': (-1, 88, 1),
    '/MBConvBlock_9/QuantConv_0/act': (-1, 91, 1),
    '/MBConvBlock_9/QuantConv_0/weight': (-1, 90, 1),
    '/MBConvBlock_9/QuantConv_1/act': (-1, 93, 0),
    '/MBConvBlock_9/QuantConv_1/weight': (-1, 92, 1),
    '/MBConvBlock_9/depthwise_conv2d/act': (-1, 95, 1),
    '/MBConvBlock_9/depthwise_conv2d/weight': (-1, 94, 1),
    '/MBConvBlock_10/QuantConv_0/act': (-1, 13, 1),
    '/MBConvBlock_10/QuantConv_0/weight': (-1, 12, 1),
    '/MBConvBlock_10/QuantConv_1/act': (-1, 15, 0),
    '/MBConvBlock_10/QuantConv_1/weight': (-1, 14, 1),
    '/MBConvBlock_10/depthwise_conv2d/act': (-1, 17, 1),
    '/MBConvBlock_10/depthwise_conv2d/weight': (-1, 16, 1),
    '/MBConvBlock_11/QuantConv_0/act': (-1, 19, 1),
    '/MBConvBlock_11/QuantConv_0/weight': (-1, 18, 1),
    '/MBConvBlock_11/QuantConv_1/act': (-1, 21, 0),
    '/MBConvBlock_11/QuantConv_1/weight': (-1, 20, 1),
    '/MBConvBlock_11/depthwise_conv2d/act': (-1, 23, 1),
    '/MBConvBlock_11/depthwise_conv2d/weight': (-1, 22, 1),
    '/MBConvBlock_12/QuantConv_0/act': (-1, 25, 1),
    '/MBConvBlock_12/QuantConv_0/weight': (-1, 24, 1),
    '/MBConvBlock_12/QuantConv_1/act': (-1, 27, 0),
    '/MBConvBlock_12/QuantConv_1/weight': (-1, 26, 1),
    '/MBConvBlock_12/depthwise_conv2d/act': (-1, 29, 1),
    '/MBConvBlock_12/depthwise_conv2d/weight': (-1, 28, 1),
    '/MBConvBlock_13/QuantConv_0/act': (-1, 31, 1),
    '/MBConvBlock_13/QuantConv_0/weight': (-1, 30, 1),
    '/MBConvBlock_13/QuantConv_1/act': (-1, 33, 0),
    '/MBConvBlock_13/QuantConv_1/weight': (-1, 32, 1),
    '/MBConvBlock_13/depthwise_conv2d/act': (-1, 35, 1),
    '/MBConvBlock_13/depthwise_conv2d/weight': (-1, 34, 1),
    '/MBConvBlock_14/QuantConv_0/act': (-1, 37, 1),
    '/MBConvBlock_14/QuantConv_0/weight': (-1, 36, 1),
    '/MBConvBlock_14/QuantConv_1/act': (-1, 39, 0),
    '/MBConvBlock_14/QuantConv_1/weight': (-1, 38, 1),
    '/MBConvBlock_14/depthwise_conv2d/act': (-1, 41, 1),
    '/MBConvBlock_14/depthwise_conv2d/weight': (-1, 40, 1),
    '/MBConvBlock_15/QuantConv_0/act': (-1, 43, 1),
    '/MBConvBlock_15/QuantConv_0/weight': (-1, 42, 1),
    '/MBConvBlock_15/QuantConv_1/act': (-1, 45, 0),
    '/MBConvBlock_15/QuantConv_1/weight': (-1, 44, 1),
    '/MBConvBlock_15/depthwise_conv2d/act': (-1, 47, 1),
    '/MBConvBlock_15/depthwise_conv2d/weight': (-1, 46, 1),
    '/head_conv/act': (-1, 97, 1),
    '/head_conv/weight': (-1, 96, 1),
    '/QuantDense_0/act': (-1, 100, 0),
    '/QuantDense_0/weight': (-1, 99, 1),
    '/QuantDense_0/bias': (-1, 108, 1),
}


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
    '/InvertedResidual_9/QuantConv_1/act': (-1, 76, 0),
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
  rng = random.PRNGKey(0)
  if config.model == 'MobileNetV2_100':
    model_cls = getattr(mbnet_model, config.model)
  else:
    model_cls = getattr(enet_models, config.model)
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

  if config.model == 'EfficientNetB0':
    enet_bits = copy.deepcopy(enet_template)
  else:
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


def plot_bits_enet(ax, config: str, workdir: str):

  if config == 'EfficientNet-Lite0':
    # try:
    #   with open('/Users/clemens/Desktop/enet0_bits.pkl', 'rb') as f:
    #     enet_bits = pickle.load(f)
    #   with open('/Users/clemens/Desktop/enet0_max.pkl', 'rb') as f:
    #     max_data = pickle.load(f)
    # except:
    enet_bits, max_data = load_data(enet_conf.get_config(), workdir)
    with open('/Users/clemens/Desktop/enet0_bits.pkl', 'wb') as f:
      pickle.dump(enet_bits, f)
    with open('/Users/clemens/Desktop/enet0_max.pkl', 'wb') as f:
      pickle.dump(max_data, f)
  if config == 'MobileNetV2':
    # try:
    #   with open('/Users/clemens/Desktop/mbnet_bits.pkl', 'rb') as f:
    #     enet_bits = pickle.load(f)
    #   with open('/Users/clemens/Desktop/mbnet_max.pkl', 'rb') as f:
    #     max_data = pickle.load(f)
    # except:
    enet_bits, max_data = load_data(mb_conf.get_config(), workdir)
    with open('/Users/clemens/Desktop/mbnet_bits.pkl', 'wb') as f:
      pickle.dump(enet_bits, f)
    with open('/Users/clemens/Desktop/mbnet_max.pkl', 'wb') as f:
      pickle.dump(max_data, f)

  def plot_fig_aw(ax, name):

    font_size = 23

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

    localw_max_bits = 7
    ax.plot([0, 108 + 1], [2 / 7, 2 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 108 + 1], [4 / 7, 4 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 108 + 1], [6 / 7, 6 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    locala_max_bits = 7
    ax.plot([0, 108 + 1], [-2 / 7, -2 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 108 + 1], [-4 / 7, -4 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)
    ax.plot([0, 108 + 1], [-6 / 7, -6 / 7], color='grey', alpha=.5,
            linestyle='--', linewidth=3, zorder=0)

    a_pos = 3
    w_pos = 1
    # extra_entry = False
    for k, v in enet_bits.items():
      if 'bias' not in k:

        if name == 'EfficientNet-Lite0':
          if ('QuantConv' in k) or ('head_conv' in k):
            label = 'Pointwise Conv.'
            color = 'orange'
          elif ('stem_conv' in k):
            label = '3x3 Convolution'
            color = 'b'
          elif ('QuantDense' in k):
            label = 'Affine'
            color = 'r'
          elif ('depthwise_conv2d' in k):
            label = 'Depthwise Conv.'
            color = 'g'
          else:
            raise Exception('Unknown error type: ' + k)
        else:
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

    ax.text(-11, 0.3, 'W # Bits', dict(size=23), rotation=90)
    ax.text(-11, -.6, 'A # Bits', dict(size=23), rotation=90)
    ax.annotate('', xy=(0, 0), xytext=(108 + 2, 0), arrowprops=dict(
        arrowstyle='<-, head_width=.3, head_length=1.', lw=3), zorder=20)
    ax.annotate('', xy=(0, -.97), xytext=(0, 1.15), arrowprops=dict(
        arrowstyle='<->, head_width=.3,  head_length=1.', lw=3), zorder=20)

    ax.set_ylim(-.97, 1.15)
    ax.set_yticks([-6 / 7, -4 / 7, -2 / 7, 0., 2 / 7, 4 / 7, 6 / 7])
    ax.set_yticklabels([6, 4, 2, 0, 2, 4, 6])
    ax.text(37, -.82, name, dict(size=23))

    handles, labels = ax.get_legend_handles_labels()

    ax.axes.xaxis.set_visible(False)

  plot_fig_aw(ax, config)


def main(argv):

  plt.rc('font', family='Helvetica', weight='bold')
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14.4, 6.5))

  plot_bits_enet(ax[0], "EfficientNet-Lite0",
                 'gs://imagenet_clemens/frontier/enet_sur/efficientnet-lite0_'
                 + 'mixed_2.4_sur_finetune_7/best')
  plot_bits_enet(ax[1], "MobileNetV2",
                 'gs://imagenet_clemens/frontier/mbnet_sur/mbnetv2_mixed_2.4_'
                 + 'finetune_7/best')

  handles, labels = ax[0].get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  fig.legend(by_label.values(), by_label.keys(),
             bbox_to_anchor=(.02, .96, .95, 0.11),
             loc="lower left", borderaxespad=0,
             ncol=4, mode='expand',
             frameon=False,
             prop={'weight': 'bold', 'size': 23})

  plt.tight_layout()
  plt.savefig('figures/bits_overview.png', dpi=300, bbox_inches='tight')
  plt.close()


if __name__ == '__main__':
  app.run(main)
