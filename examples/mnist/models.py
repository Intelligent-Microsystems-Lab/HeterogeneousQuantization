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

sys.path.append("efficientnet")
from enet_load_pretrained_weights import enet_load_pretrained_weights  # noqa: E402, E501
from efficientnet_utils import BlockDecoder, GlobalParams  # noqa: E402

sys.path.append("..")
from flax_qconv import QuantConv  # noqa: E402
from flax_qdense import QuantDense  # noqa: E402
from batchnorm import BatchNorm  # noqa: E402

ModuleDef = Any
Array = Any


def fake_quant_conv(config, bits, quant_act_sign, g_scale, **kwargs):
  return nn.Conv(**kwargs)


conv_kernel_initializer = partial(variance_scaling, 1.0, "fan_in", "normal")
dense_kernel_initializer = partial(
    variance_scaling, 1.0 / 3.0, "fan_out", "uniform")


class MnistNet(nn.Module):
  """EfficientNet."""
  depth: int
  num_classes: int
  dtype: Any = jnp.float32
  act: Callable = jax.nn.relu6
  config: dict = ml_collections.FrozenConfigDict({})
  load_model_fn: Callable = enet_load_pretrained_weights

  @nn.compact
  def __call__(self, x: Array, train: bool = True, rng: Any = None) -> Array:
    # Default parameters from efficientnet lite builder
    if self.config.quant.bits is None:
      conv = partial(fake_quant_conv, dtype=self.dtype,
                     g_scale=self.config.quant.g_scale)
    else:
      conv = partial(QuantConv, dtype=self.dtype,
                     g_scale=self.config.quant.g_scale)
    norm = partial(BatchNorm,
                   use_running_average=not train,
                   dtype=self.dtype,
                   use_bias=True,
                   use_scale=True)
  
    self.sow('intermediates', 'inputs', x)
    for i in range(self.depth):
      x = conv(
        features=2**(i+4),
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='SAME',
        kernel_init=conv_kernel_initializer(),
        use_bias=False,
        config=self.config.quant.mbconv,
        bits=self.config.quant.bits,
        quant_act_sign=True,)(x)

      x = norm()(x)
      x = self.act(x)
      self.sow('intermediates', 'stem'+str(i), x)
      logging.info('Built layer with output shape: %s', x.shape)

    x = jnp.mean(x, axis=(1, 2))
    
    if self.config.quant.bits is None:
      x = nn.Dense(self.num_classes,
                   kernel_init=dense_kernel_initializer(),
                   dtype=self.dtype)(x)
    else:
      x = QuantDense(self.num_classes,
                     kernel_init=dense_kernel_initializer(),
                     dtype=self.dtype,
                     config=self.config.quant.dense,
                     bits=self.config.quant.bits,
                     quant_act_sign=False,
                     g_scale=self.config.quant.g_scale)(x)
    x = jnp.asarray(x, self.dtype)
    self.sow('intermediates', 'head', x)

    return x


MnistNetB0 = partial(MnistNet, depth=3)

