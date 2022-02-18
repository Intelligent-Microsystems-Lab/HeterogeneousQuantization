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
    '3': '0',
    '4': '1',
    '6': '2',
    '7': '3',
    '9': '4',
    '10': '5',
    '11': '6',
    '12': '7',
    'squeeze': 'QuantConv_0',
    'expand1x1': 'QuantConv_1',
    'expand3x3': 'QuantConv_2',
}


def squeezenet_load_pretrained_weights(state, location):

  if '.pth' in location:
    # Load torch style squeezenet.
    torch_state = torch.load(location, map_location=torch.device('cpu'))

    torch_weights = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.params['params']))

    for key, value in torch_state.items():

      if key == 'classifier.1.weight':
        torch_weights['head_conv']['kernel'] = jnp.moveaxis(
            jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
        continue
      if key == 'classifier.1.bias':
        torch_weights['head_conv']['bias'] = jnp.array(value)
        continue

      if 'features' in key:
        key_parts = key.split('.')

        # stem conv params
        if key_parts[1] == '0' and key_parts[2] == 'weight':
          torch_weights['stem_conv']['kernel'] = jnp.moveaxis(
              jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
          continue
        elif key_parts[1] == '0' and key_parts[2] == 'bias':
          torch_weights['stem_conv']['bias'] = jnp.array(value)
          continue
        else:
          if key_parts[-1] == 'weight':
            torch_weights['Fire_' + map_dict[key_parts[1]]][map_dict[
                key_parts[2]]]['kernel'] = jnp.moveaxis(
                jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
            continue
          else:
            torch_weights['Fire_' + map_dict[key_parts[1]]
                          ][map_dict[key_parts[2]]]['bias'] = jnp.array(value)
            continue

    general_params = {'params': torch_weights,
                      'quant_params': state.params['quant_params']}
    batch_stats = state.batch_stats
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

  return TrainState.create(
      apply_fn=state.apply_fn,
      params=general_params,
      tx=state.tx,
      batch_stats=freeze(batch_stats),
      weight_size=state.weight_size,
      act_size=state.act_size,
  )
