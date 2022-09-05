# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
from flax.core import freeze, unfreeze
from flax.training import train_state
from flax.training import checkpoints

import jax.numpy as jnp
import numpy as np

from typing import Any

import torch


class TrainState(train_state.TrainState):
  batch_stats: Any
  weight_size: Any
  act_size: Any


def sum_list_of_tensors(x):
  return np.sum([y.sum() for y in x])


# torch to jax map.
map_dict = {
    '0': 'stem',
    '1': '0',
    '2': '1',
    '3': '2',
    '4': '3',
    '5': '4',
    '6': '5',
    '7': '6',
    '8': '7',
    '9': '8',
    '10': '9',
    '11': '10',
    '12': '11',
    '13': '12',
    '14': '13',
    '15': '14',
    '16': '15',
    '17': '16',
    '18': 'head'
}


def mobilenetv2_load_pretrained_weights(state, location):

  if '.pth' in location:
    # Load torch style mobilenetv2.
    torch_state = torch.load(location, map_location=torch.device('cpu'))

    torch_weights = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.params['params']))
    torch_bn_stats = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.batch_stats))

    for key, value in torch_state.items():
      if 'features' in key:
        key_parts = key.split('.')

        if (map_dict[key_parts[1]] == 'stem') or (
                map_dict[key_parts[1]] == 'head'):
          # conv params
          if key_parts[2] == '0':
            torch_weights[map_dict[key_parts[1]] + '_conv'
                          ]['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue

          # batch norm params
          if key_parts[2] == '1':
            if key_parts[-1] == 'weight':
              torch_weights[map_dict[key_parts[1]]
                            + '_bn']['scale'] = jnp.array(value)
              continue
            if key_parts[-1] == 'bias':
              torch_weights[map_dict[key_parts[1]]
                            + '_bn']['bias'] = jnp.array(value)
              continue

          # batch stats
          if key_parts[-1] == 'running_mean':
            torch_bn_stats[map_dict[key_parts[1]]
                           + '_bn']['mean'] = jnp.array(value)
            continue
          if key_parts[-1] == 'running_var':
            torch_bn_stats[map_dict[key_parts[1]]
                           + '_bn']['var'] = jnp.array(value)
            continue

        if map_dict[key_parts[1]] == '0':
          # only two conv layer
          if key == 'features.1.conv.0.0.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['QuantConv_0']['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue
          if key == 'features.1.conv.0.1.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_0']['scale'] = jnp.array(value)
            continue
          if key == 'features.1.conv.0.1.bias':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_0']['bias'] = jnp.array(value)
            continue
          if key == 'features.1.conv.0.1.running_mean':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_0']['mean'] = jnp.array(value)
            continue
          if key == 'features.1.conv.0.1.running_var':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_0']['var'] = jnp.array(value)
            continue
          if key == 'features.1.conv.1.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['QuantConv_1']['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue
          if key == 'features.1.conv.2.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_1']['scale'] = jnp.array(value)
            continue
          if key == 'features.1.conv.2.bias':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_1']['bias'] = jnp.array(value)
            continue
          if key == 'features.1.conv.2.running_mean':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_1']['mean'] = jnp.array(value)
            continue
          if key == 'features.1.conv.2.running_var':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_1']['var'] = jnp.array(value)
            continue
        else:
          key_end = key.split('.conv.')

          if key_end[-1] == '0.0.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['QuantConv_0']['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue
          if key_end[-1] == '0.1.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_0']['scale'] = jnp.array(value)
            continue
          if key_end[-1] == '0.1.bias':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_0']['bias'] = jnp.array(value)
            continue
          if key_end[-1] == '0.1.running_mean':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_0']['mean'] = jnp.array(value)
            continue
          if key_end[-1] == '0.1.running_var':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_0']['var'] = jnp.array(value)
            continue

          if key_end[-1] == '1.0.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['QuantConv_1']['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue
          if key_end[-1] == '1.1.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_1']['scale'] = jnp.array(value)
            continue
          if key_end[-1] == '1.1.bias':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_1']['bias'] = jnp.array(value)
            continue
          if key_end[-1] == '1.1.running_mean':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_1']['mean'] = jnp.array(value)
            continue
          if key_end[-1] == '1.1.running_var':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_1']['var'] = jnp.array(value)
            continue

          if key_end[-1] == '2.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['QuantConv_2']['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue
          if key_end[-1] == '3.weight':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_2']['scale'] = jnp.array(value)
            continue
          if key_end[-1] == '3.bias':
            torch_weights['InvertedResidual_' + map_dict[key_parts[1]]
                          ]['BatchNorm_2']['bias'] = jnp.array(value)
            continue
          if key_end[-1] == '3.running_mean':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_2']['mean'] = jnp.array(value)
            continue
          if key_end[-1] == '3.running_var':
            torch_bn_stats['InvertedResidual_' + map_dict[key_parts[1]]
                           ]['BatchNorm_2']['var'] = jnp.array(value)
            continue

      if key == 'classifier.1.weight':
        torch_weights['QuantDense_0']['kernel'] = jnp.array(value).transpose()
        continue
      if key == 'classifier.1.bias':
        torch_weights['QuantDense_0']['bias'] = jnp.array(value)
        continue

    general_params = {'params': torch_weights,
                      'quant_params': state.params['quant_params']}
    batch_stats = torch_bn_stats
    quant_params = None
  else:

    chk_state = checkpoints.restore_checkpoint(location, None)
    chk_weights, _ = jax.tree_util.tree_flatten(chk_state['params']['params'])
    _, weight_def = jax.tree_util.tree_flatten(state.params['params'])
    params = jax.tree_util.tree_unflatten(weight_def, chk_weights)

    chk_batchstats, _ = jax.tree_util.tree_flatten(chk_state['batch_stats'])
    _, batchstats_def = jax.tree_util.tree_flatten(state.batch_stats)
    batch_stats = jax.tree_util.tree_unflatten(batchstats_def, chk_batchstats)

    general_params = {'params': params,
                      'quant_params': state.params['quant_params']}

    if 'quant_params' in chk_state['params']:
      quant_params = chk_state['params']['quant_params']
    else:
      quant_params = None

  return quant_params, TrainState.create(
      apply_fn=state.apply_fn,
      params=general_params,
      tx=state.tx,
      batch_stats=freeze(batch_stats),
      weight_size=state.weight_size,
      act_size=state.act_size,
  )
