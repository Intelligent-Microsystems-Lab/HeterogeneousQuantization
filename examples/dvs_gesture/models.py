

import sys

import functools
from functools import partial
import ml_collections
from typing import Any, Callable, Sequence, Tuple, Union, Iterable

import jax
from flax import linen as nn
from flax.linen.initializers import lecun_normal
import jax.numpy as jnp


from jax._src.nn.initializers import variance_scaling

conv_kernel_initializer = partial(variance_scaling, 1.0, "fan_in", "normal")
dense_kernel_initializer = partial(
    variance_scaling, 1.0 / 3.0, "fan_out", "uniform")


@jax.custom_vjp
def spike_ste(x, thr):
  return jnp.array(x>thr, dtype=bool)


def spike_ste_fwd(x, thr):
  return spike_ste(x, thr), (x, thr)


def spike_ste_bwd(res, g):
  (x, thr) = res

  return g, None


spike_ste.defvjp(spike_ste_fwd, spike_ste_bwd)


class SpikingBlock(nn.Module):
  connection_fn: Callable
  norm_fn: Callable
  spike_fn: Callable
  alpha: float = .75
  threshold: float = 1.0
  dtype: Any = jnp.float32

  @functools.partial(
      nn.transforms.scan,
      variable_broadcast='params',
      split_rngs={'params': False})
  @nn.compact
  def __call__(self, u, inputs):

    x = self.connection_fn(inputs)
    #x = self.norm_fn()(x)

    u = self.alpha * u + x
    out = self.spike_fn(u, self.threshold)
    u -= out

    return u, out

  @staticmethod
  def initialize_carry(inputs, connection_fn):
    x = connection_fn(inputs[0,:])
    return jnp.zeros(x.shape, jnp.float32)


class SpikingConvNet(nn.Module):
  """SpikingConvNet."""
  stage_sizes: Sequence[int]
  num_filters: int = 64
  num_classes: int = 11
  dtype: Any = jnp.float32
  t_collapse_fn: Callable = lambda x: x.mean(axis=0)
  config: dict = ml_collections.FrozenConfigDict({})

  
  @nn.compact
  def __call__(self, inputs, train: bool = True, rng: Any = None):

    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.99,
                   epsilon=1e-3,
                   dtype=jnp.float32,
                   use_bias=True,
                   use_scale=True)

    # Stem part.
    stem_conv = SpikingBlock(
          connection_fn = nn.Conv( features=self.num_filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            kernel_init=conv_kernel_initializer(),
            use_bias=False,
            name='stem_conv',
          ),
          norm_fn = norm,
          spike_fn = spike_ste,
        )
    init_carry = stem_conv.initialize_carry(inputs, stem_conv.connection_fn)
    _, x = stem_conv(init_carry, inputs)

    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        conv_block = SpikingBlock(
          connection_fn = nn.Conv( features=self.num_filters * 2 ** i,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='SAME',
            kernel_init=conv_kernel_initializer(),
            use_bias=False,
            name='block_' + str(i) + '_' + str(j),
          ),
          norm_fn = norm,
          spike_fn = spike_ste,
        )
        init_carry = conv_block.initialize_carry(x, conv_block.connection_fn)
        _, x = conv_block(init_carry, x)
        

    # x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
    head_dense = SpikingBlock(
          connection_fn = nn.Dense(
            self.num_classes,
            kernel_init=dense_kernel_initializer(),
            dtype=self.dtype,
            name='head'
          ),
          norm_fn = norm,
          spike_fn = spike_ste,
        )
    init_carry = head_dense.initialize_carry(x, head_dense.connection_fn)
    _, x = head_dense(init_carry, x)
    x = jnp.asarray(x, self.dtype)
    
    return self.t_collapse_fn(x)



SpikingConvNet0 = partial(SpikingConvNet, [1,1,1])
# SpikingConvNet1 = partial(SpikingConvNet, width_coefficient=1.0,
#                          depth_coefficient=1.1,
#                          dropout_rate=0.2)
# SpikingConvNet2 = partial(SpikingConvNet, width_coefficient=1.1,
#                          depth_coefficient=1.2,
#                          dropout_rate=0.3)
# SpikingConvNet3 = partial(SpikingConvNet, width_coefficient=1.2,
#                          depth_coefficient=1.4,
#                          dropout_rate=0.3)
# SpikingConvNet4 = partial(SpikingConvNet, width_coefficient=1.4,
#                          depth_coefficient=1.8,
#                          dropout_rate=0.3)
