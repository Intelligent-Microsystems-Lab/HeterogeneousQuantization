# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
# and https://github.com/tensorflow/tpu/blob/master/models/official/ \
# efficientnet/efficientnet_model.py

"""Flax implementation of EfficientNet"""

# See issue #620.
# pytype: disable=wrong-arg-count

import sys
from functools import partial
import ml_collections
from typing import Any, Callable, Tuple

from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import math
from absl import logging

import jax
from jax._src.nn.initializers import variance_scaling

from efficientnet_utils import BlockDecoder, GlobalParams

sys.path.append("../..")
from flax_qconv import QuantConv  # noqa: E402
from flax_qdense import QuantDense  # noqa: E402

ModuleDef = Any
Array = Any


_DEFAULT_BLOCKS_ARGS = [
    'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
    'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
    'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
    'r1_k3_s11_e6_i192_o320_se0.25',
]

conv_kernel_initializer = partial(variance_scaling, 1.0, "fan_in", "normal")
dense_kernel_initializer = partial(
    variance_scaling, 1.0 / 3.0, "fan_out", "uniform")


def round_filters(filters: int,
                  width_coefficient: float,
                  depth_divisor: float,
                  min_depth: float,
                  skip: bool = False) -> int:
  """Round number of filters based on depth multiplier."""
  orig_f = filters
  multiplier = width_coefficient
  divisor = depth_divisor
  min_depth = min_depth
  if skip or not multiplier:
    return filters

  filters *= multiplier
  min_depth = min_depth or divisor
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  logging.info('round_filter input=%s output=%s', orig_f, new_filters)
  return int(new_filters)


def round_repeats(repeats: int,
                  depth_coefficient: float,
                  skip: bool = False) -> int:
  """Round number of filters based on depth multiplier."""
  multiplier = depth_coefficient
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, is_training, survival_prob, rng):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = inputs.shape[0]
  random_tensor = survival_prob
  random_tensor += jax.random.uniform(rng,
                                      [batch_size, 1, 1, 1],
                                      dtype=inputs.dtype)
  binary_tensor = jnp.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = inputs / survival_prob * binary_tensor
  return output


class MBConvBlock(nn.Module):
  """EfficientNet block."""
  conv: ModuleDef
  norm: ModuleDef
  expand_ratio: float
  input_filters: int
  kernel_size: Tuple
  strides: Tuple
  output_filters: int
  clip_projection_output: bool
  id_skip: bool
  act: Callable
  config: dict
  bits: int
  block_num: int

  @nn.compact
  def __call__(self,
               inputs: Array,
               rng: Any,
               train: bool = True,
               survival_prob: float = None) -> Array:
    logging.info('Block %s input shape: %s', self.name, inputs.shape)
    x = inputs

    # Otherwise, first apply expansion and then apply depthwise conv.
    if self.expand_ratio != 1:
      filters = self.input_filters * self.expand_ratio
      kernel_size = self.kernel_size

      x = self.conv(features=filters,
                    kernel_size=[1, 1],
                    strides=[1, 1],
                    kernel_init=conv_kernel_initializer(),
                    padding='SAME',
                    use_bias=False,
                    config=self.config,
                    bits=self.bits,
                    quant_act_sign=True)(x)
      x = self.norm()(x)
      x = self.act(x)
      logging.info('Expand shape: %s', x.shape)

    # Depthwise convolution
    kernel_size = self.kernel_size
    feature_group_count = x.shape[-1]
    features = int(1 * feature_group_count)

    x = self.conv(
        features=features,
        kernel_size=[kernel_size, kernel_size],
        strides=self.strides,
        kernel_init=conv_kernel_initializer(),
        padding='SAME',
        feature_group_count=feature_group_count,
        use_bias=False,
        name='depthwise_conv2d',
        config=self.config,
        bits=self.bits,
        quant_act_sign=True)(x)
    x = self.norm()(x)
    x = self.act(x)
    logging.info('DWConv shape: %s', x.shape)

    filters = self.output_filters
    x = self.conv(features=filters,
                  kernel_size=[1, 1],
                  strides=[1, 1],
                  kernel_init=conv_kernel_initializer(),
                  padding='SAME',
                  use_bias=False,
                  config=self.config,
                  bits=self.bits,
                  quant_act_sign=False)(x)
    x = self.norm()(x)

    if self.clip_projection_output:
      x = jnp.clip(x, a_min=-6, a_max=6)
    if self.id_skip:
      if all(
          s == 1 for s in self.strides
      ) and inputs.shape[-1] == x.shape[-1]:
        # Apply only if skip connection presents.
        if survival_prob:
          # this is supposed to be drop connect
          x = drop_connect(x, train, survival_prob, rng)
        x = x + inputs
    logging.info('Project shape: %s', x.shape)
    return x


def get_survival_prob(survival_prob: float,
                      idx: int,
                      len_blocks: int) -> float:
  if survival_prob:
    drop_rate = 1.0 - survival_prob
    survival_prob = 1.0 - drop_rate * float(idx) / len_blocks
    logging.info('block_%s survival_prob: %s', idx, survival_prob)
  return survival_prob


class EfficientNet(nn.Module):
  """EfficientNet."""
  width_coefficient: float
  depth_coefficient: float
  resolution: int
  dropout_rate: float
  num_classes: int
  dtype: Any = jnp.float32
  act: Callable = jax.nn.relu6
  config: dict = ml_collections.FrozenConfigDict({})

  @nn.compact
  def __call__(self, x: Array, train: bool = True, rng: Any = None) -> Array:
    # Default parameters from efficientnet lite builder
    global_params = GlobalParams(
        blocks_args=_DEFAULT_BLOCKS_ARGS,
        batch_norm_momentum=0.99,
        batch_norm_epsilon=self.config.batch_norm_epsilon if hasattr(
            self.config, 'batch_norm_epsilon') else 1e-3,
        dropout_rate=self.dropout_rate,
        survival_prob=.8,
        width_coefficient=self.width_coefficient,
        depth_coefficient=self.depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        relu_fn=jax.nn.relu6,
        clip_projection_output=False,
        fix_head_stem=True,  # Don't scale stem and head.
        local_pooling=True,  # special cases for tflite issues.
        use_se=False)

    _blocks_args = BlockDecoder().decode(global_params.blocks_args)

    conv = partial(QuantConv, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=global_params.batch_norm_momentum,
                   epsilon=global_params.batch_norm_epsilon,
                   dtype=self.dtype,
                   use_bias=True,
                   use_scale=True)
    rfliters = partial(round_filters,
                       width_coefficient=global_params.width_coefficient,
                       depth_divisor=global_params.depth_divisor,
                       min_depth=global_params.min_depth,)
    conv_block = partial(
        MBConvBlock,
        clip_projection_output=global_params.clip_projection_output,
        config=self.config.quant.mbconv, bits=self.config.quant.bits)

    self.sow('intermediates', 'inputs', x)
    # Stem part.
    x = conv(
        features=rfliters(32, skip=global_params.fix_head_stem),
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='SAME',
        kernel_init=conv_kernel_initializer(),
        use_bias=False,
        name='stem_conv',
        config=self.config.quant.stem,
        bits=self.config.quant.bits)(x)

    x = norm(name='stem_bn')(x)
    x = self.act(x)
    self.sow('intermediates', 'stem', x)
    logging.info('Built stem layers with output shape: %s', x.shape)

    # Builds blocks.
    idx = 0
    total_num_blocks = np.sum([x.num_repeat for x in _blocks_args])
    for i, block_args in enumerate(_blocks_args):
      assert block_args.num_repeat > 0
      assert block_args.space2depth in [0, 1, 2]

      # Update block input and output filters based on depth multiplier.
      input_filters = rfliters(block_args.input_filters)
      output_filters = rfliters(block_args.output_filters)

      if (i == 0 or i == len(_blocks_args) - 1):
        repeats = block_args.num_repeat
      else:
        repeats = round_repeats(block_args.num_repeat, self.depth_coefficient)

      block_args = block_args._replace(
          input_filters=input_filters,
          output_filters=output_filters,
          num_repeat=repeats)

      if not block_args.space2depth:
        survival_prob = get_survival_prob(
            global_params.survival_prob, idx, total_num_blocks)
        rng, prng = jax.random.split(rng, 2)
        x = conv_block(conv=conv,
                       norm=norm,
                       expand_ratio=block_args.expand_ratio,
                       input_filters=block_args.input_filters,
                       kernel_size=block_args.kernel_size,
                       strides=block_args.strides,
                       output_filters=block_args.output_filters,
                       id_skip=block_args.id_skip,
                       act=self.act,
                       block_num=idx)(x, train=train,
                                      survival_prob=survival_prob, rng=prng)
        idx += 1
      else:
        assert False, 'Space2depth is not implemented.'

      if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
        # pylint: disable=protected-access
        block_args = block_args._replace(
            input_filters=block_args.output_filters, strides=[1, 1])
        # pylint: enable=protected-access
      for _ in range(block_args.num_repeat - 1):
        survival_prob = get_survival_prob(
            global_params.survival_prob, idx, total_num_blocks)
        rng, prng = jax.random.split(rng, 2)
        x = conv_block(conv=conv,
                       norm=norm,
                       expand_ratio=block_args.expand_ratio,
                       input_filters=block_args.input_filters,
                       kernel_size=block_args.kernel_size,
                       strides=block_args.strides,
                       output_filters=block_args.output_filters,
                       id_skip=block_args.id_skip,
                       act=self.act,
                       block_num=idx)(x, train=train,
                                      survival_prob=survival_prob, rng=prng)
        idx += 1

      if i == 0 or i == 5:
        self.sow('intermediates', 'features' + str(i), x)

    # Head part.
    x = conv(features=rfliters(1280, skip=global_params.fix_head_stem),
             kernel_size=(1, 1),
             strides=(1, 1),
             padding='SAME',
             kernel_init=conv_kernel_initializer(),
             use_bias=False,
             name='head_conv',
             config=self.config.quant.head,
             bits=self.config.quant.bits,)(x)
    x = norm(name='head_bn')(x)
    x = self.act(x)

    x = jnp.mean(x, axis=(1, 2))
    if 'average' in self.config.quant:
      x = self.config.quant.average()(x, sign=False)

    x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
    x = QuantDense(self.num_classes,
                   kernel_init=dense_kernel_initializer(),
                   dtype=self.dtype,
                   config=self.config.quant.dense,
                   bits=self.config.quant.bits,
                   quant_act_sign=False)(x)
    x = jnp.asarray(x, self.dtype)
    self.sow('intermediates', 'head', x)

    return x


EfficientNetB0 = partial(EfficientNet, width_coefficient=1.0,
                         depth_coefficient=1.0, resolution=224,
                         dropout_rate=0.2)
EfficientNetB1 = partial(EfficientNet, width_coefficient=1.0,
                         depth_coefficient=1.1, resolution=240,
                         dropout_rate=0.2)
EfficientNetB2 = partial(EfficientNet, width_coefficient=1.1,
                         depth_coefficient=1.2, resolution=260,
                         dropout_rate=0.3)
EfficientNetB3 = partial(EfficientNet, width_coefficient=1.2,
                         depth_coefficient=1.4, resolution=280,
                         dropout_rate=0.3)
EfficientNetB4 = partial(EfficientNet, width_coefficient=1.4,
                         depth_coefficient=1.8, resolution=300,
                         dropout_rate=0.3)
