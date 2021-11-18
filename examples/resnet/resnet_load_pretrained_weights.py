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
import h5py


class TrainState(train_state.TrainState):
  batch_stats: Any
  weight_size: Any
  act_size: Any


map_dict = {
    '10': '0',
    '11': '1',
    '12': '2',
    '20': '3',
    '21': '4',
    '22': '5',
    '30': '6',
    '31': '7',
    '32': '8',
}

map_dict_nnabla = {
    '11': '0',
    '12': '1',
    '13': '2',
    '21': '3',
    '22': '4',
    '23': '5',
    '31': '6',
    '32': '7',
    '33': '8',
}


def resnet_load_pretrained_weights(state, location):

  # Note JAX convolutions are H(eight)W(idth)I(n)O(ut)
  # NNABLA convolutions are O(ut)I(n)H(eight)W(idth)

  if '.h5' in location:

    nnabla_weights = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.params['params']))
    nnabla_bn_stats = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.batch_stats))

    h5f = h5py.File(location, 'r')
    value_list = []

    def add_tolist(name):
      value_list.append(name)
    h5f.visit(add_tolist)
    for key in value_list:
      if 'basicblock' in key and 'quantized_conv' in key and 'W' in key:
        num1 = key.split('/')[0][-1]
        num2 = key.split('/')[1][-1]
        num3 = int(key.split('/')[2][-1])

        values = h5f[key][:]
        values = jnp.moveaxis(values, (0, 1, 2, 3), (3, 2, 0, 1))

        target_shape = nnabla_weights['ResNetBlock_' + map_dict_nnabla[
            num1 + num2]]['QuantConv_' + str(num3 - 1)]['kernel'].shape

        assert values.shape == target_shape, 'Weight Shape missmatch at ' + \
            key + ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['ResNetBlock_' + map_dict_nnabla[num1 + num2]
                       ]['QuantConv_' + str(num3 - 1)]['kernel'] = values
        continue
      if 'basicblock' in key and 'bn' in key and 'beta' in key:
        num1 = key.split('/')[0][-1]
        num2 = key.split('/')[1][-1]
        num3 = int(key.split('/')[2][-1])

        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_weights['ResNetBlock_' + map_dict_nnabla[
            num1 + num2]]['BatchNorm_' + str(num3 - 1)]['bias'].shape

        assert values.shape == target_shape, 'Bias Shape missmatch at ' + \
            key + ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['ResNetBlock_' + map_dict_nnabla[num1 + num2]
                       ]['BatchNorm_' + str(num3 - 1)]['bias'] = values
        continue

      if 'basicblock' in key and 'bn' in key and 'gamma' in key:
        num1 = key.split('/')[0][-1]
        num2 = key.split('/')[1][-1]
        num3 = int(key.split('/')[2][-1])

        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_weights['ResNetBlock_' + map_dict_nnabla[
            num1 + num2]]['BatchNorm_' + str(num3 - 1)]['scale'].shape

        assert values.shape == target_shape, 'Scale Shape missmatch at ' + \
            key + ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['ResNetBlock_' + map_dict_nnabla[num1 + num2]
                       ]['BatchNorm_' + str(num3 - 1)]['scale'] = values
        continue

      if 'basicblock' in key and 'bn' in key and 'mean' in key:
        num1 = key.split('/')[0][-1]
        num2 = key.split('/')[1][-1]
        num3 = int(key.split('/')[2][-1])

        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_bn_stats['ResNetBlock_' + map_dict_nnabla[
            num1 + num2]]['BatchNorm_' + str(num3 - 1)]['mean'].shape

        assert values.shape == target_shape, 'Mean Shape missmatch at ' + \
            key + ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_bn_stats['ResNetBlock_' + map_dict_nnabla[num1 + num2]
                        ]['BatchNorm_' + str(num3 - 1)]['mean'] = values

        continue

      if 'basicblock' in key and 'bn' in key and 'var' in key:
        num1 = key.split('/')[0][-1]
        num2 = key.split('/')[1][-1]
        num3 = int(key.split('/')[2][-1])

        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_bn_stats['ResNetBlock_' + map_dict_nnabla[
            num1 + num2]]['BatchNorm_' + str(num3 - 1)]['var'].shape

        assert values.shape == target_shape, 'Mean Shape missmatch at ' + \
            key + ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_bn_stats['ResNetBlock_' + map_dict_nnabla[num1 + num2]
                        ]['BatchNorm_' + str(num3 - 1)]['var'] = values
        continue

      # affine weight
      if 'fc/quantized_affine/W' in key:
        values = h5f[key][:]

        target_shape = nnabla_weights['QuantDense_0']['kernel'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['QuantDense_0']['kernel'] = values
        continue

      # assine bias
      if 'fc/quantized_affine/b' in key:
        values = h5f[key][:]

        target_shape = nnabla_weights['QuantDense_0']['bias'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['QuantDense_0']['bias'] = values
        continue

      # conv1 weight
      if 'conv1/quantized_conv/W' in key:
        values = h5f[key][:]
        values = jnp.moveaxis(values, (0, 1, 2, 3), (3, 2, 0, 1))

        target_shape = nnabla_weights['conv_init']['kernel'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['conv_init']['kernel'] = values
        continue

      # bn1 gamma
      if 'conv1/bn/gamma' in key:
        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_weights['bn_init']['scale'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['bn_init']['scale'] = values
        continue

      # bn1 beta
      if 'conv1/bn/beta' in key:
        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_weights['bn_init']['bias'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_weights['bn_init']['bias'] = values
        continue

      # bn1 mean
      if 'conv1/bn/mean' in key:
        values = h5f[key][0, :, 0, 0]
        target_shape = nnabla_bn_stats['bn_init']['mean'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_bn_stats['bn_init']['mean'] = values
        continue

      # bn1 var
      if 'conv1/bn/var' in key:
        values = h5f[key][0, :, 0, 0]

        target_shape = nnabla_bn_stats['bn_init']['var'].shape

        assert values.shape == target_shape, 'Shape missmatch at ' + key + \
            ' NNABLA shape: ' + str(values.shape) + \
            ' JAX shape: ' + str(target_shape)

        nnabla_bn_stats['bn_init']['var'] = values
        continue

    general_params = {'params': nnabla_weights,
                      'quant_params': state.params['quant_params']}
    batch_stats = nnabla_bn_stats

  elif '.th' in location:
    torch_state = torch.load(location, map_location=torch.device('cpu'))

    torch_weights = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.params['params']))
    torch_bn_stats = unfreeze(jax.tree_util.tree_map(
        lambda x: jnp.zeros(x.shape), state.batch_stats))
    for key, value in torch_state['state_dict'].items():
      if 'module.layer' in key:
        layer_num = int(key.split('.')[1][-1])
        sub_layer_num = int(key.split('.')[2])
        if key.split('.')[3] == 'conv1':
          torch_weights['ResNetBlock_' + map_dict[str(layer_num) + str(
              sub_layer_num)]]['QuantConv_0']['kernel'] = jnp.moveaxis(
              jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
          continue

        if key.split('.')[3] == 'conv2':
          torch_weights['ResNetBlock_' + map_dict[str(layer_num) + str(
              sub_layer_num)]]['QuantConv_1']['kernel'] = jnp.moveaxis(
              jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
          continue

        if key.split('.')[3] == 'bn1':
          if key.split('.')[4] == 'weight':
            torch_weights['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_0']['scale'] = jnp.array(value)
            continue

          if key.split('.')[4] == 'bias':
            torch_weights['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_0']['bias'] = jnp.array(value)
            continue

          if key.split('.')[4] == 'running_mean':
            torch_bn_stats['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_0']['mean'] = jnp.array(value)
            continue

          if key.split('.')[4] == 'running_var':
            torch_bn_stats['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_0']['var'] = jnp.array(value)
            continue

        if key.split('.')[3] == 'bn2':
          if key.split('.')[4] == 'weight':
            torch_weights['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_1']['scale'] = jnp.array(value)
            continue

          if key.split('.')[4] == 'bias':
            torch_weights['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_1']['bias'] = jnp.array(value)
            continue

          if key.split('.')[4] == 'running_mean':
            torch_bn_stats['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_1']['mean'] = jnp.array(value)
            continue

          if key.split('.')[4] == 'running_var':
            torch_bn_stats['ResNetBlock_' + map_dict[str(layer_num) + str(
                sub_layer_num)]]['BatchNorm_1']['var'] = jnp.array(value)
            continue

      if key == 'module.conv1.weight':
        torch_weights['conv_init']['kernel'] = jnp.moveaxis(
            jnp.array(value), (0, 1, 2, 3), (3, 2, 0, 1))
        continue
      if key == 'module.bn1.weight':
        torch_weights['bn_init']['scale'] = jnp.array(value)
        continue
      if key == 'module.bn1.bias':
        torch_weights['bn_init']['bias'] = jnp.array(value)
        continue
      if key == 'module.linear.weight':
        torch_weights['QuantDense_0']['kernel'] = jnp.array(value).transpose()
        continue
      if key == 'module.linear.bias':
        torch_weights['QuantDense_0']['bias'] = jnp.array(value)
        continue

      if key == 'module.bn1.running_mean':
        torch_bn_stats['bn_init']['mean'] = jnp.array(value)
        continue
      if key == 'module.bn1.running_var':
        torch_bn_stats['bn_init']['var'] = jnp.array(value)
        continue

    def sum_list_of_tensors(x):
      return np.sum([y.sum() for y in x])

    assert int(sum_list_of_tensors(
        jax.tree_util.tree_flatten(
            torch_bn_stats)[0]) * 1000 + sum_list_of_tensors(
        jax.tree_util.tree_flatten(torch_weights)[0]) * 1000) == int(
        sum_list_of_tensors(torch_state['state_dict'].values()) * 1000)
    general_params = {'params': torch_weights,
                      'quant_params': state.params['quant_params']}
    batch_stats = torch_bn_stats
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
