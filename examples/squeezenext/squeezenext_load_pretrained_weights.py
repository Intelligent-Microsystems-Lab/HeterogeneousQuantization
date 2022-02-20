# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
from flax.core import freeze, unfreeze
from flax.training import train_state
from flax.training import checkpoints

import jax.numpy as jnp
import numpy as np

from typing import Any

from tf2cv.model_provider import get_model as tf2cv_get_model


class TrainState(train_state.TrainState):
  batch_stats: Any
  weight_size: Any
  act_size: Any


def sum_list_of_tensors(x):
  return np.sum([y.sum() for y in x])


# torch to jax map.
map_tf_to_jax = {
    # 6 6 8 1
    '11': '0',
    '12': '1',
    '13': '2',
    '14': '3',
    '15': '4',
    '16': '5',
    '21': '6',
    '22': '7',
    '23': '8',
    '24': '9',
    '25': '10',
    '26': '11',
    '31': '12',
    '32': '13',
    '33': '14',
    '34': '15',
    '35': '16',
    '36': '17',
    '37': '18',
    '38': '19',
    '41': '20',
}


def squeezenext_load_pretrained_weights(state, location):

  if 'imgclsmob' in location:
    tf_net = tf2cv_get_model(location.split(
        '-')[1], pretrained=True, data_format="channels_last")

    tf_weights = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.params['params']))
    tf_bn_stats = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.batch_stats))

    for value in tf_net.variables:

      if value.name == 'output1/kernel:0':
        tf_weights['QuantDense_0']['kernel'] = jnp.array(value.numpy())
        continue
      if value.name == 'output1/bias:0':
        tf_weights['QuantDense_0']['bias'] = jnp.array(value.numpy())
        continue

      key_parts = value.name.split('/')

      if (key_parts[2] == 'init_block'):
        if key_parts[4] == 'conv':
          if 'kernel' in key_parts[-1]:
            assert tf_weights['stem_conv']['kernel'].shape == value.shape
            tf_weights['stem_conv']['kernel'] = jnp.array(value.numpy())
            continue
          if 'bias' in key_parts[-1]:
            tf_weights['stem_conv']['bias'] = jnp.array(value.numpy())
            continue

        if key_parts[4] == 'bn':
          if 'gamma' in key_parts[-1]:
            tf_weights['BatchNorm_0']['scale'] = jnp.array(value.numpy())
            continue
          if 'beta' in key_parts[-1]:
            tf_weights['BatchNorm_0']['bias'] = jnp.array(value.numpy())
            continue
          if 'mean' in key_parts[-1]:
            tf_bn_stats['BatchNorm_0']['mean'] = jnp.array(value.numpy())
            continue
          if 'variance' in key_parts[-1]:
            tf_bn_stats['BatchNorm_0']['var'] = jnp.array(value.numpy())
            continue

      if (key_parts[2] == 'final_block'):
        if key_parts[3] == 'conv':
          if 'kernel' in key_parts[-1]:
            tf_weights['head_conv']['kernel'] = jnp.array(value.numpy())
            continue
          if 'bias' in key_parts[-1]:
            tf_weights['head_conv']['bias'] = jnp.array(value.numpy())
            continue
        if key_parts[3] == 'bn':
          if 'gamma' in key_parts[-1]:
            tf_weights['BatchNorm_1']['scale'] = jnp.array(value.numpy())
            continue
          if 'beta' in key_parts[-1]:
            tf_weights['BatchNorm_1']['bias'] = jnp.array(value.numpy())
            continue
          if 'mean' in key_parts[-1]:
            tf_bn_stats['BatchNorm_1']['mean'] = jnp.array(value.numpy())
            continue
          if 'variance' in key_parts[-1]:
            tf_bn_stats['BatchNorm_1']['var'] = jnp.array(value.numpy())
            continue

      if 'stage' in key_parts[2]:
        unit = map_tf_to_jax[key_parts[3].split(
            'stage')[1] + key_parts[4].split('unit')[1]]

        if key_parts[5] == 'identity_conv':
          conv = '5'
        else:
          conv = str(int(key_parts[5].split('conv')[1]) - 1)

        if key_parts[6] == 'conv':
          if 'kernel' in key_parts[-1]:
            tf_weights['SqnxtUnit_'
                       + unit]['QuantConv_'
                               + conv]['kernel'] = jnp.array(value.numpy())
            continue
          if 'bias' in key_parts[-1]:
            tf_weights['SqnxtUnit_'
                       + unit]['QuantConv_'
                               + conv]['bias'] = jnp.array(value.numpy())
            continue
        if key_parts[6] == 'bn':
          if 'gamma' in key_parts[-1]:
            tf_weights['SqnxtUnit_'
                       + unit]['BatchNorm_'
                               + conv]['scale'] = jnp.array(value.numpy())
            continue
          if 'beta' in key_parts[-1]:
            tf_weights['SqnxtUnit_'
                       + unit]['BatchNorm_'
                               + conv]['bias'] = jnp.array(value.numpy())
            continue
          if 'mean' in key_parts[-1]:
            tf_bn_stats['SqnxtUnit_'
                        + unit]['BatchNorm_'
                                + conv]['mean'] = jnp.array(value.numpy())
            continue
          if 'variance' in key_parts[-1]:
            tf_bn_stats['SqnxtUnit_'
                        + unit]['BatchNorm_'
                                + conv]['var'] = jnp.array(value.numpy())
            continue

    general_params = {'params': tf_weights,
                      'quant_params': state.params['quant_params']}
    batch_stats = tf_bn_stats
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
