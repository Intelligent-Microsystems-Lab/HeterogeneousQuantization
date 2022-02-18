# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples and
# https://pytorch.org/vision/stable/_modules/torchvision/models/squeezenet.html

"""Flax implementation of MobileNetV2."""
import sys

from functools import partial
import ml_collections
from typing import Any, Callable
from absl import logging

from flax import linen as nn
from jax.nn.initializers import normal, kaiming_uniform
import jax.numpy as jnp

sys.path.append('squeezenet')
from squeezenet_load_pretrained_weights import squeezenet_load_pretrained_weights  # noqa: E402, E501

sys.path.append("..")
from flax_qconv import QuantConv  # noqa: E402

ModuleDef = Any
Array = Any


class Fire(nn.Module):
  """Fire block."""
  squeeze_planes: int
  expand1x1_planes: int
  expand3x3_planes: int
  conv: ModuleDef
  width_mult: float
  config: dict = ml_collections.FrozenConfigDict({})

  @nn.compact
  def __call__(self, x: Array, train: bool = True,) -> Array:
    logging.info('%s input shape: %s', self.name, x.shape)

    # squeeze
    x = self.conv(int(self.squeeze_planes * self.width_mult), (1, 1), padding=(
        (0, 0), (0, 0)), config=self.config)(x)
    x = nn.relu(x)

    # expand1x1
    x_1x1 = self.conv(int(self.expand1x1_planes * self.width_mult), (1, 1), padding=(
        (0, 0), (0, 0)), config=self.config)(x)
    x_1x1 = nn.relu(x_1x1)

    # expand3x3
    x_3x3 = self.conv(int(self.expand1x1_planes * self.width_mult), (3, 3), padding=(
        (1, 1), (1, 1)), config=self.config)(x)
    x_3x3 = nn.relu(x_3x3)

    return jnp.concatenate([x_1x1, x_3x3], axis=3)


class SqueezeNet(nn.Module):
  """SqueezeNet 1.1."""
  num_classes: int = 1000
  width_mult: float = 1.0
  config: dict = ml_collections.FrozenConfigDict({})
  load_model_fn: Callable = squeezenet_load_pretrained_weights
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: Array, train: bool = True, rng: Any = None) -> Array:
    conv = partial(QuantConv, use_bias=True, dtype=self.dtype,
                   bias_init=nn.initializers.zeros,
                   bits=self.config.quant.bits,
                   kernel_init=kaiming_uniform(),)
    fire = partial(Fire, conv = conv, width_mult = self.width_mult, config=self.config.quant.fire)
    _ = self.variable('batch_stats', 'placeholder', lambda x: 0., (1,))

    # Building first layer.
    logging.info('Stem input shape: %s', x.shape)
    x = conv(features=int(64*self.width_mult), kernel_size=(3, 3), strides=(2, 2),
             padding=((0, 0), (0, 0)),
             name='stem_conv',
             config=self.config.quant.stem,
             bits=self.config.quant.bits,
             )(x)
    x = nn.relu(x)
    self.sow('intermediates', 'stem', x)

    # Building Fire blocks.
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')
    x = fire(16, 64, 64)(x)
    self.sow('intermediates', 'fire_0', x)
    x = fire(16, 64, 64)(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')
    self.sow('intermediates', 'fire_1', x)
    x = fire(32, 128, 128)(x)
    self.sow('intermediates', 'fire_2', x)
    x = fire(32, 128, 128)(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')
    self.sow('intermediates', 'fire_3', x)
    x = fire(48, 192, 192)(x)
    self.sow('intermediates', 'fire_4', x)
    x = fire(48, 192, 192)(x)
    self.sow('intermediates', 'fire_5', x)
    x = fire(64, 256, 256)(x)
    self.sow('intermediates', 'fire_6', x)
    x = fire(64, 256, 256)(x)
    self.sow('intermediates', 'fire_7', x)

    x = nn.Dropout(0.5)(x, deterministic=not train)

    # Building classifier.
    logging.info('Head input shape: %s', x.shape)
    x = QuantConv(self.num_classes, (1, 1), strides=(1, 1),
                  name='head_conv', padding=((0, 0), (0, 0)),
                  kernel_init=normal(0.01),
                  bias_init=nn.initializers.zeros,
                  use_bias=True,
                  dtype=self.dtype,
                  config=self.config.quant.head,
                  bits=self.config.quant.bits)(x)
    x = nn.relu(x)
    x = jnp.mean(x, axis=(1, 2))

    x = jnp.asarray(x, self.dtype)

    return x


WideSqueezeNet = partial(SqueezeNet, width_mult=2.0)
