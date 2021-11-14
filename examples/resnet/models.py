# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count
import sys

from functools import partial
import ml_collections
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp


sys.path.append("../..")
from flax_qconv import QuantConv  # noqa: E402
from flax_qdense import QuantDense  # noqa: E402
from batchnorm import BatchNorm

ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  config: dict = ml_collections.FrozenConfigDict({})
  bits: int = 8
  shortcut_type: str = 'B'

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides, config=self.config,
                  quant_act_sign=False, bits=self.bits)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), config=self.config, bits = self.bits,
                  quant_act_sign=False)(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      if self.shortcut_type == 'A':
        if residual.shape[-1] != y.shape[-1]:
          # pad channel with zero
          zero_channel_pad = jnp.zeros(residual.shape[:-1] + (y.shape[-1] - residual.shape[-1],))
          residual = jnp.concatenate((residual, zero_channel_pad), 3)
          # residual = jnp.pad(residual[:, ::2, ::2, :], ((0, 0), (0, 0), (0, 0), (self.filters//4, self.filters//4)), "constant", constant_values = 0)
        if self.strides != (1,1):
         # this removes excess entries
         residual = nn.avg_pool(residual, (1, 1), strides=self.strides, padding='SAME')
      else: # Option B
        residual = self.conv(self.filters, (1, 1),
                             self.strides, name='conv_proj', config=self.config,
                             bits = self.bits, quant_act_sign=False)(residual)
        residual = self.norm(name='norm_proj')(residual)
    return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)
  config: dict = ml_collections.FrozenConfigDict({})
  bits: int = 8
  shortcut_type: str = 'B'

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1), config=self.config, bits = self.bits,
                  quant_act_sign=False)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides, config=self.config,
                  bits = self.bits, quant_act_sign=False)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1), config=self.config, bits = self.bits,
                  quant_act_sign=False)(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj', config=self.config,
                           bits = self.bits, quant_act_sign=False)(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  config: dict = ml_collections.FrozenConfigDict({})
  shortcut_type: str = 'B'

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(QuantConv, use_bias=False, dtype=self.dtype)
    norm = partial(BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)
    if x.shape[-3:]  == (32, 32, 3):
      cifar10_flag = True
      kernel_size = (3,3)
      stride_size = (1,1)
      padding_size = [(1,1),(1,1)]
    else:
      cifar10_flag = False
      kernel_size = (7,7)
      stride_size = (2,2)
      padding_size = [(3,3),(3,3)]
    x = jnp.ones_like(x)
    x = conv(self.num_filters, kernel_size, stride_size, padding=padding_size, name='conv_init', config=self.config.quant.stem, bits=self.config.quant.bits, quant_act_sign=False)(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    if not cifar10_flag:
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act,
                           config=self.config.quant.mbconv,
                           bits=self.config.quant.bits,
                           shortcut_type=self.shortcut_type)(x)
      
    if cifar10_flag and 'average' in self.config.quant:
      x = self.config.quant.average(bits = self.config.quant.bits)(x, sign=False)

    x = jnp.mean(x, axis=(1, 2))
    if not cifar10_flag and 'average' in self.config.quant:
      x = self.config.quant.average(bits = self.config.quant.bits)(x, sign=False)
    x = QuantDense(self.num_classes, dtype=self.dtype,
                   config=self.config.quant.dense, quant_act_sign=False, bits=self.config.quant.bits)(x)
    x = jnp.asarray(x, self.dtype)
    return x


ResNet20_CIFAR10 = partial(ResNet, stage_sizes=[3, 3, 3],
                   block_cls=ResNetBlock, num_filters = 16, shortcut_type = 'A')

ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2],
                   block_cls=ResNetBlock)
ResNet34 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=ResNetBlock)
ResNet50 = partial(ResNet, stage_sizes=[3, 4, 6, 3],
                   block_cls=BottleneckResNetBlock)
ResNet101 = partial(ResNet, stage_sizes=[3, 4, 23, 3],
                    block_cls=BottleneckResNetBlock)
ResNet152 = partial(ResNet, stage_sizes=[3, 8, 36, 3],
                    block_cls=BottleneckResNetBlock)
ResNet200 = partial(ResNet, stage_sizes=[3, 24, 36, 3],
                    block_cls=BottleneckResNetBlock)


# Used for testing only.
_ResNet1 = partial(ResNet, stage_sizes=[1], block_cls=ResNetBlock)
