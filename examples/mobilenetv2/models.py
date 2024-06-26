# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples and
# https://pytorch.org/vision/stable/\
# _modules/torchvision/models/mobilenetv2.html

"""Flax implementation of MobileNetV2."""
import sys

from functools import partial
import ml_collections
from typing import Any, Tuple, Optional, Callable
from absl import logging

import jax
from flax import linen as nn
from jax.nn.initializers import variance_scaling, normal
import jax.numpy as jnp

sys.path.append('mobilenetv2')
from mobilenetv2_load_pretrained_weights import mobilenetv2_load_pretrained_weights  # noqa: E402, E501

sys.path.append("..")
from flax_qconv import QuantConv  # noqa: E402
from flax_qdense import QuantDense  # noqa: E402
from batchnorm import BatchNorm  # noqa: E402

ModuleDef = Any
Array = Any
default_kernel_init = partial(
    variance_scaling, 2.0, "fan_out", "truncated_normal")


def _make_divisible(v, divisor, min_value=None):
  """
  Originally copied from
  https://pytorch.org/vision/0.8/_modules/torchvision/models/mobilenet.html

  This function is taken from the original tf repo.
  It ensures that all layers have a channel number that is divisible by 8
  It can be seen here:
  https://github.com/tensorflow/models/blob/master/\
  research/slim/nets/mobilenet/mobilenet.py
  :param v:
  :param divisor:
  :param min_value:
  :return:
  """
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


class InvertedResidual(nn.Module):
  """InvertedResidual block."""
  oup: int
  stride: int
  expand_ratio: int
  norm: ModuleDef
  conv: ModuleDef
  config: dict = ml_collections.FrozenConfigDict({})
  bits: int = 8

  @nn.compact
  def __call__(self, x: Array, train: bool = True,) -> Array:
    logging.info('%s input shape: %s', self.name, x.shape)
    residual = x
    assert self.stride in [(1, 1), (2, 2)]

    hidden_dim = int(round(x.shape[3] * self.expand_ratio))

    if self.expand_ratio != 1:
      # pw
      x = self.conv(hidden_dim, (1, 1), padding=(
          (0, 0), (0, 0)), config=self.config, bits=self.bits)(x)
      x = self.norm()(x)
      x = jax.nn.relu6(x)

    # dw
    x = self.conv(hidden_dim, (3, 3), strides=self.stride,
                  padding=((1, 1), (1, 1)), config=self.config,
                  feature_group_count=hidden_dim, bits=self.bits,
                  quant_act_sign=False)(x)
    x = self.norm()(x)
    x = jax.nn.relu6(x)

    # pw-linear
    x = self.conv(self.oup, (1, 1), padding=(
        (0, 0), (0, 0)), config=self.config, bits=self.bits,
        quant_act_sign=False)(x)
    x = self.norm()(x)

    if self.stride == (1, 1) and residual.shape[3] == self.oup:
      logging.info('use_res_connect: True')
      x += residual

    return x


class MobileNetV2(nn.Module):
  """MobileNetV2."""
  num_classes: int = 1000
  width_mult: float = 1.0
  inverted_residual_setting: Optional[Tuple[Tuple[int]]] = (
      # t, c, n, s
      (1, 16, 1, (1, 1)),
      (6, 24, 2, (2, 2)),
      (6, 32, 3, (2, 2)),
      (6, 64, 4, (2, 2)),
      (6, 96, 3, (1, 1)),
      (6, 160, 3, (2, 2)),
      (6, 320, 1, (1, 1)),
  )
  round_nearest: int = 8
  config: dict = ml_collections.FrozenConfigDict({})
  load_model_fn: Callable = mobilenetv2_load_pretrained_weights
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x: Array, train: bool = True, rng: Any = None) -> Array:

    conv = partial(QuantConv, use_bias=False, dtype=self.dtype,
                   bits=self.config.quant.bits,
                   kernel_init=default_kernel_init(),
                   g_scale=self.config.quant.g_scale)
    norm = partial(BatchNorm,
                   scale_init=nn.initializers.ones,
                   bias_init=nn.initializers.zeros,
                   use_running_average=not train,
                   momentum=.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    # only check the first element, assuming user knows t,c,n,s are required
    if len(self.inverted_residual_setting) == 0 or len(
            self.inverted_residual_setting[0]) != 4:
      raise ValueError("inverted_residual_setting should be non-empty "
                       "or a 4-element list, got {}".format(
                           self.inverted_residual_setting))

    input_channel = 32
    last_channel = 1280

    # building first layer
    input_channel = _make_divisible(
        input_channel * self.width_mult, self.round_nearest)
    last_channel = _make_divisible(
        last_channel * max(1.0, self.width_mult), self.round_nearest)

    logging.info('Stem input shape: %s', x.shape)
    x = conv(features=input_channel, kernel_size=(3, 3), strides=(2, 2),
             padding=((1, 1), (1, 1)),
             name='stem_conv',
             config=self.config.quant.stem,
             bits=self.config.quant.bits,
             )(x)
    x = norm(name='stem_bn')(x)
    x = jax.nn.relu6(x)
    self.sow('intermediates', 'stem', x)

    # building inverted residual blocks
    block_counter = 0
    for t, c, n, s in self.inverted_residual_setting:
      output_channel = _make_divisible(c * self.width_mult, self.round_nearest)
      logging.info('InvertedResidual sequence')
      for i in range(n):
        stride = s if i == 0 else (1, 1)
        x = InvertedResidual(
            oup=output_channel,
            stride=stride,
            expand_ratio=t,
            norm=norm,
            conv=conv,
            config=self.config.quant.invertedresidual,
            bits=self.config.quant.bits)(x)
        self.sow('intermediates', 'features_' + str(block_counter), x)
        block_counter += 1

    # building last several layers
    logging.info('Head input shape: %s', x.shape)
    x = conv(last_channel, (1, 1), strides=(1, 1),
             name='head_conv', padding=((0, 0), (0, 0)),
             config=self.config.quant.head,
             bits=self.config.quant.bits)(x)
    x = norm(name='head_bn')(x)
    x = jax.nn.relu6(x)
    self.sow('intermediates', 'head', x)

    logging.info('Average pool input shape: %s', x.shape)
    x = jnp.mean(x, axis=(1, 2))

    x = nn.Dropout(0.2)(x, deterministic=not train)

    x = QuantDense(self.num_classes, dtype=self.dtype,
                   config=self.config.quant.dense,
                   bits=self.config.quant.bits,
                   kernel_init=normal(0.01),
                   bias_init=nn.initializers.zeros,
                   quant_act_sign=False,
                   g_scale=self.config.quant.g_scale
                   )(x)
    x = jnp.asarray(x, self.dtype)

    return x


# https://arxiv.org/pdf/1801.04381.pdf
MobileNetV2_140 = partial(MobileNetV2, width_mult=1.4)
MobileNetV2_130 = partial(MobileNetV2, width_mult=1.3)
MobileNetV2_100 = partial(MobileNetV2, width_mult=1.0)
MobileNetV2_075 = partial(MobileNetV2, width_mult=0.75)
MobileNetV2_050 = partial(MobileNetV2, width_mult=0.5)
MobileNetV2_035 = partial(MobileNetV2, width_mult=0.35)
