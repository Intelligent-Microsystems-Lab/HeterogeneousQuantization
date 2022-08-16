# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

from absl import app

import pickle
import copy

import jax
from jax import random
import jax.numpy as jnp
import numpy as np

import ml_collections
from flax.core import unfreeze

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

# @clee1994 update sign bits!!!!
enet_template = {
    '/stem_conv': [-1, 1, 8, 1, ],
    '/MBConvBlock_0/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_0/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_1/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_1/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_1/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_2/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_2/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_2/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_3/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_3/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_3/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_4/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_4/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_4/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_5/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_5/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_5/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_6/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_6/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_6/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_7/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_7/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_7/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_8/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_8/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_8/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_9/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_9/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_9/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_10/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_10/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_10/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_11/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_11/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_11/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_12/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_12/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_12/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_13/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_13/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_13/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_14/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_14/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_14/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/MBConvBlock_15/QuantConv_0': [-1, 1, -1, 1, ],
    '/MBConvBlock_15/QuantConv_1': [-1, 1, -1, 1, ],
    '/MBConvBlock_15/depthwise_conv2d': [-1, 1, -1, 1, ],
    '/head_conv': [-1, 1, -1, 1, ],
    '/QuantDense_0': [-1, 1, -1, 1, ],
}

mbnet_template = {
    '/stem_conv': [-1, 1, 8, 1, ],
    '/InvertedResidual_0/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_0/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_1/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_1/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_1/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_2/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_2/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_2/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_3/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_3/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_3/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_4/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_4/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_4/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_5/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_5/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_5/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_6/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_6/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_6/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_7/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_7/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_7/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_8/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_8/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_8/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_9/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_9/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_9/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_10/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_10/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_10/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_11/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_11/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_11/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_12/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_12/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_12/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_13/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_13/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_13/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_14/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_14/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_14/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_15/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_15/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_15/QuantConv_2': [-1, 1, -1, 1, ],
    '/InvertedResidual_16/QuantConv_0': [-1, 1, -1, 1, ],
    '/InvertedResidual_16/QuantConv_1': [-1, 1, -1, 1, ],
    '/InvertedResidual_16/QuantConv_2': [-1, 1, -1, 1, ],
    '/head_conv': [-1, 1, -1, 1, ],
    '/QuantDense_0': [-1, 1, -1, 1, ],
}

# slack number estimates from Genys MAC synthesis
max_freq = {
    11: 1010.1,
    12: 891.27,
    13: 820.34,
    14: 852.51,
    15: 801.92,
    16: 757,
    17: 716.85,
    18: 680.74,
    21: 891.27,
    22: 782.47,
    23: 708.22,
    24: 646.83,
    25: 595.24,
    26: 551.27,
    27: 535.62,
    28: 495.05,
    31: 820.34,
    32: 708.22,
    33: 639.8,
    34: 587.54,
    35: 585.82,
    36: 537.35,
    37: 496.03,
    38: 461.25,
    41: 852.51,
    42: 646.83,
    43: 587.54,
    44: 591.37,
    45: 543.18,
    46: 501.25,
    47: 465.12,
    48: 434.4,
    51: 801.92,
    52: 595.24,
    53: 585.82,
    54: 542.3,
    55: 505.56,
    56: 469.7,
    57: 437.83,
    58: 410.34,
    61: 757,
    62: 551.27,
    63: 537.35,
    64: 500.5,
    65: 469.04,
    66: 441.31,
    67: 396.51,
    68: 373.83,
    71: 716.85,
    72: 535.62,
    73: 496.03,
    74: 464.47,
    75: 437.25,
    76: 395.88,
    77: 391.24,
    78: 356,
    81: 680.74,
    82: 495.05,
    83: 461.25,
    84: 433.65,
    85: 409.84,
    86: 373.41,
    87: 355.62,
    88: 339.33,
}

mac_latency = {11: 4010,
               12: 3878,
               13: 3781,
               14: 3827,
               15: 3753,
               16: 3679,
               17: 3605,
               18: 3531,
               21: 3878,
               22: 3722,
               23: 3588,
               24: 3454,
               25: 3320,
               26: 3186,
               27: 3133,
               28: 2980,
               31: 3781,
               32: 3588,
               33: 3437,
               34: 3298,
               35: 3293,
               36: 3139,
               37: 2984,
               38: 2832,
               41: 3827,
               42: 3454,
               43: 3298,
               44: 3309,
               45: 3159,
               46: 3005,
               47: 2850,
               48: 2698,
               51: 3753,
               52: 3320,
               53: 3293,
               54: 3156,
               55: 3022,
               56: 2871,
               57: 2716,
               58: 2563,
               61: 3679,
               62: 3186,
               63: 3139,
               64: 3002,
               65: 2868,
               66: 2734,
               67: 2478,
               68: 2325,
               71: 3605,
               72: 3133,
               73: 2984,
               74: 2847,
               75: 2713,
               76: 2474,
               77: 2444,
               78: 2191,
               81: 3531,
               82: 2980,
               83: 2832,
               84: 2694,
               85: 2560,
               86: 2322,
               87: 2188,
               88: 2053, }


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

  if config.model == 'EfficientNetB0':
    enet_bits = copy.deepcopy(enet_template)
  else:
    enet_bits = copy.deepcopy(mbnet_template)

  # Forumla for MACs in Conv
  # (K^2) * C_in * H_out * W_out * C_out
  # from https://stackoverflow.com/questions/56138754/
  # formula-to-compute-the-number-of-macs-in-a-convolutional-neural-network
  # Formula for MACs in Dense

  for i in flat_params:
    if i == '/placeholder':
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '0':
      if enet_bits[i.split('/parametric_d_xmax_')[0]][0] == -1:
        # weights
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(
            state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0]][1]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_weights[i.split(
            '/parametric_d_xmax_0')[0] + '/parametric_d_xmax_0/weight_mb'] \
            / bits_comp

        enet_bits[i.split('/parametric_d_xmax_')[0]][0] = np.max(bits_comp)
        enet_bits[i.split('/parametric_d_xmax_')[0]
                  ][1] = np.max(num_params)  # or ACE metric later
      continue
    if i.split('/parametric_d_xmax_')[1][0] == '1':
      if enet_bits[i.split('/parametric_d_xmax_')[0]][2] == -1:
        # act
        path = i[1:].split('/')[:-1]
        d = fetch_value(state.params['quant_params'], path + ['step_size'])
        xmax = fetch_value(
            state.params['quant_params'], path + ['dynamic_range'])
        sign_bit = enet_bits[i.split('/parametric_d_xmax_')[0]][3]
        bits_comp = jnp.ceil(jnp.log2(xmax / d)) + sign_bit
        num_params = flat_acts[i.split(
            '/parametric_d_xmax_1')[0] + '/parametric_d_xmax_1/act_mb'] \
            / bits_comp

        enet_bits[i.split('/parametric_d_xmax_')[0]][2] = np.max(bits_comp)
        enet_bits[i.split('/parametric_d_xmax_')[0]
                  ][3] = np.max(num_params)  # or ACE metric later
      continue

  return enet_bits


def main(argv):

  config = 'MobileNetV2'  # EfficientNet-Lite0'
  workdir = '/Users/clemens/Desktop/mixed_models/mbnet7'

  if config == 'EfficientNet-Lite0':
    # try:
    #   with open('/Users/clemens/Desktop/enet0_bits.pkl', 'rb') as f:
    #     enet_bits = pickle.load(f)
    #   with open('/Users/clemens/Desktop/enet0_max.pkl', 'rb') as f:
    #     max_data = pickle.load(f)
    # except:
    enet_bits = load_data(enet_conf.get_config(), workdir)
    with open('/Users/clemens/Desktop/enet_bits.pkl', 'wb') as f:
      pickle.dump(enet_bits, f)
    # with open('/Users/clemens/Desktop/enet0_max.pkl', 'wb') as f:
    #   pickle.dump(max_data, f)
  if config == 'MobileNetV2':
    # try:
    #   with open('/Users/clemens/Desktop/mbnet_bits.pkl', 'rb') as f:
    #     enet_bits = pickle.load(f)
    #   with open('/Users/clemens/Desktop/mbnet_max.pkl', 'rb') as f:
    #     max_data = pickle.load(f)
    # except:
    enet_bits = load_data(mb_conf.get_config(), workdir)
    with open('/Users/clemens/Desktop/mbnet_bits.pkl', 'wb') as f:
      pickle.dump(enet_bits, f)
    # with open('/Users/clemens/Desktop/mbnet_max.pkl', 'wb') as f:
    #  pickle.dump(max_data, f)

  lat_num = 0
  for k, i in enet_bits.items():
    lat_num += 1 / max_freq[int(i[0]) * 10 + int(i[2])]
    print(k + str(' ') + str(int(i[0]) * 10 + int(i[2])))
  print(np.round(lat_num * 1000, 2))

  # lat_num = 0
  # for k,i in enet_bits.items():
  #   lat_num += 1/200
  #   # print(k + str(' ') + str(int(i[0]) * 10 + int(i[2])))
  # print(np.round(lat_num*1000, 2))


if __name__ == '__main__':
  app.run(main)
