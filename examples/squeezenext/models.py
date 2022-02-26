# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples and
# https://github.com/osmr/imgclsmob/blob/master/tensorflow2/tf2cv/\
# models/squeezenext.py

"""Flax implementation of SqueezeNext."""
import sys

from functools import partial
import ml_collections
from typing import Any, Callable, Sequence
from absl import logging

from flax import linen as nn
import jax.numpy as jnp

sys.path.append('squeezenext')
from squeezenext_load_pretrained_weights import squeezenext_load_pretrained_weights  # noqa: E402, E501

sys.path.append("..")
from flax_qconv import QuantConv  # noqa: E402
from flax_qdense import QuantDense  # noqa: E402
from batchnorm import BatchNorm  # noqa: E402

ModuleDef = Any
Array = Any


class SqnxtUnit(nn.Module):
  """SqnxtUnit."""
  out_channels: int
  strides: int
  conv: ModuleDef
  norm: ModuleDef
  config: dict = ml_collections.FrozenConfigDict({})

  @nn.compact
  def __call__(self, x: Array, train: bool = True,) -> Array:
    logging.info('%s input shape: %s', self.name, x.shape)

    in_channels = x.shape[-1]
    residual = x

    if self.strides == (2, 2):
      reduction_den = 1
      resize_identity = True
    elif in_channels > self.out_channels:
      reduction_den = 4
      resize_identity = True
    else:
      reduction_den = 2
      resize_identity = False

    x = self.conv(in_channels // reduction_den, (1, 1), strides=self.strides,
                  padding=((0, 0), (0, 0)), config=self.config,
                  quant_act_sign=False)(x)
    x = self.norm()(x)
    x = nn.relu(x)

    x = self.conv(in_channels // (reduction_den * 2), (1, 1), strides=(1, 1),
                  padding=((0, 0), (0, 0)), config=self.config,
                  quant_act_sign=False)(x)
    x = self.norm()(x)
    x = nn.relu(x)

    x = self.conv(in_channels // reduction_den, (1, 3), strides=(1, 1),
                  padding=((0, 0), (1, 1)), config=self.config,
                  quant_act_sign=False)(x)
    x = self.norm()(x)
    x = nn.relu(x)

    x = self.conv(in_channels // reduction_den, (3, 1), strides=(1, 1),
                  padding=((1, 1), (0, 0)), config=self.config,
                  quant_act_sign=False)(x)
    x = self.norm()(x)
    x = nn.relu(x)

    x = self.conv(self.out_channels, (1, 1), strides=(1, 1),
                  padding=((0, 0), (0, 0)), config=self.config,
                  quant_act_sign=False)(x)
    x = self.norm()(x)
    x = nn.relu(x)

    if resize_identity:
      residual = self.conv(self.out_channels, (1, 1),
                           strides=self.strides,
                           padding=((0, 0), (0, 0)),
                           config=self.config, quant_act_sign=False)(residual)
      residual = self.norm()(residual)
      residual = nn.relu(residual)

    x += residual
    x = nn.relu(x)

    return x


class SqueezeNext(nn.Module):
  """SqueezeNext"""
  stage_sizes: Sequence[int]
  width_mult: float = 1.0
  num_classes: int = 1000
  config: dict = ml_collections.FrozenConfigDict({})
  load_model_fn: Callable = squeezenext_load_pretrained_weights
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: Array, train: bool = True, rng: Any = None) -> Array:
    conv = partial(QuantConv, use_bias=True, dtype=self.dtype,
                   bits=self.config.quant.bits,
                   g_scale=self.config.quant.g_scale,)
    norm = partial(BatchNorm,
                   use_running_average=not train,
                   momentum=.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    # Building stem.
    logging.info('Stem input shape: %s', x.shape)
    x = conv(features=int(64 * self.width_mult),
             kernel_size=(7, 7),
             strides=(2, 2),
             padding=((1, 0), (1, 0)),
             name='stem_conv',
             config=self.config.quant.stem,
             )(x)
    x = norm()(x)
    x = nn.relu(x)
    self.sow('intermediates', 'stem_conv_track', x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 0), (1, 0)))
    self.sow('intermediates', 'stem', x)

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if (j == 0) and (i != 0) else (1, 1)
        x = SqnxtUnit(out_channels=int(32 * (2**i) * self.width_mult),
                      strides=strides,
                      conv=conv,
                      norm=norm,
                      config=self.config.quant.sqnxtunit)(x)
        self.sow('intermediates', 'sqnxt_' + str(i) + '_' + str(j), x)

    # Building head.
    logging.info('Head input shape: %s', x.shape)
    x = conv(int(128 * self.width_mult),
             kernel_size=(1, 1),
             strides=(1, 1),
             padding=((0, 0), (0, 0)),
             name='head_conv',
             config=self.config.quant.head, quant_act_sign=False)(x)
    x = norm()(x)
    x = nn.relu(x)

    # Average
    x = jnp.mean(x, axis=(1, 2))
    self.sow('intermediates', 'head', x)

    # Dense
    x = QuantDense(self.num_classes,
                   dtype=self.dtype,
                   config=self.config.quant.dense,
                   bits=self.config.quant.bits,
                   quant_act_sign=False,
                   g_scale=self.config.quant.g_scale)(x)
    x = jnp.asarray(x, self.dtype)

    return x


sqnxt23_w1 = partial(SqueezeNext, stage_sizes=[6, 6, 8, 1], width_mult=1.0)
sqnxt23_w3d2 = partial(SqueezeNext, stage_sizes=[6, 6, 8, 1], width_mult=1.5)
sqnxt23_w2 = partial(SqueezeNext, stage_sizes=[6, 6, 8, 1], width_mult=2.0)
sqnxt23v5_w1 = partial(SqueezeNext, stage_sizes=[2, 4, 14, 1], width_mult=1.0)
sqnxt23v5_w3d2 = partial(SqueezeNext, stage_sizes=[
    2, 4, 14, 1], width_mult=1.5)
sqnxt23v5_w2 = partial(SqueezeNext, stage_sizes=[2, 4, 14, 1], width_mult=2.0)
