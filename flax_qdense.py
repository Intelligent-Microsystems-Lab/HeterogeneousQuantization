# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Copied code from
# https://github.com/google/flax/blob/master/flax/linen/linear.py and
# modified to accomodate noise and quantization

from typing import (
    Any,
    Callable,
    Sequence,
)


from flax.linen.module import Module, compact
from flax.linen.initializers import lecun_normal, zeros

import jax
import jax.numpy as jnp


default_kernel_init = lecun_normal()

Array = Any
Dtype = Any
PRNGKey = Any
Shape = Sequence[int]


def get_noise(x, percentage, rng):
  return (
      jnp.max(jnp.abs(x))
      * percentage
      * jax.random.uniform(rng, x.shape, minval=-1, maxval=1.0)
  )


@jax.custom_vjp
def dot_general(
    inpt,
    kernel,
    rng,
    weight_noise,
    act_noise,
    weight_bwd_noise,
    act_bwd_noise,
    err_inpt_noise,
    err_weight_noise,
):
  rng, prng = jax.random.split(rng, 2)
  kernel = kernel + get_noise(kernel, weight_noise, prng)

  rng, prng = jax.random.split(rng, 2)
  inpt = inpt + get_noise(inpt, act_noise, prng)

  return jnp.dot(inpt, kernel)


def dot_general_fwd(
    inpt,
    kernel,
    rng,
    weight_noise,
    act_noise,
    weight_bwd_noise,
    act_bwd_noise,
    err_inpt_noise,
    err_weight_noise,
):
  rng, prng = jax.random.split(rng, 2)
  return dot_general(
      inpt,
      kernel,
      rng,
      weight_noise,
      act_noise,
      weight_bwd_noise,
      act_bwd_noise,
      err_inpt_noise,
      err_weight_noise,
  ), (
      inpt,
      kernel,
      prng,
      weight_noise,
      act_noise,
      weight_bwd_noise,
      act_bwd_noise,
      err_inpt_noise,
      err_weight_noise,
  )


def dot_general_bwd(res, g):
  (
      inpt,
      kernel,
      rng,
      weight_noise,
      act_noise,
      weight_bwd_noise,
      act_bwd_noise,
      err_inpt_noise,
      err_weight_noise,
  ) = res
  g_inpt = g_weight = g

  rng, prng = jax.random.split(rng, 2)
  kernel = kernel + get_noise(kernel, weight_bwd_noise, prng)

  rng, prng = jax.random.split(rng, 2)
  inpt = inpt + get_noise(inpt, act_bwd_noise, prng)

  rng, prng = jax.random.split(rng, 2)
  g_inpt = g_inpt + get_noise(g_inpt, err_inpt_noise, prng)

  rng, prng = jax.random.split(rng, 2)
  g_weight = g_weight + get_noise(g_weight, err_weight_noise, prng)

  g_inpt_fwd = jnp.dot(g_inpt, jnp.transpose(kernel))

  g_kernel_fwd = jnp.dot(jnp.transpose(inpt), g_weight)

  return (g_inpt_fwd, g_kernel_fwd, None, None, None, None, None, None, None)


dot_general.defvjp(dot_general_fwd, dot_general_bwd)


class QuantDense(Module):
  """A linear transformation applied over the last dimension of the input.
  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    config: bit widths and other configurations
  """

  features: int
  use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  config: dict = None

  @compact
  def __call__(self, inputs: Array, rng: Any = None) -> Array:
    """Applies a linear transformation to the inputs along the last
      dimension.
    Args:
      inputs: The nd-array to be transformed.
    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param(
        "kernel", self.kernel_init, (inputs.shape[-1], self.features)
    )
    kernel = jnp.asarray(kernel, self.dtype)

    if self.config is None or "act_noise" not in self.config:
      act_noise = 0.0
    else:
      act_noise = self.config["act_noise"]

    if self.config is None or "weight_noise" not in self.config:
      weight_noise = 0.0
    else:
      weight_noise = self.config["weight_noise"]

    if self.config is None or "act_bwd_noise" not in self.config:
      act_bwd_noise = 0.0
    else:
      act_bwd_noise = self.config["act_bwd_noise"]

    if self.config is None or "weight_bwd_noise" not in self.config:
      weight_bwd_noise = 0.0
    else:
      weight_bwd_noise = self.config["weight_bwd_noise"]

    if self.config is None or "err_inpt_noise" not in self.config:
      err_inpt_noise = 0.0
    else:
      err_inpt_noise = self.config["err_inpt_noise"]

    if self.config is None or "err_weight_noise" not in self.config:
      err_weight_noise = 0.0
    else:
      err_weight_noise = self.config["err_weight_noise"]

    y = dot_general(
        inputs,
        kernel,
        rng,
        jnp.array(weight_noise),
        jnp.array(act_noise),
        jnp.array(weight_bwd_noise),
        jnp.array(act_bwd_noise),
        jnp.array(err_inpt_noise),
        jnp.array(err_weight_noise),
    )
    if self.use_bias:
      bias = self.param("bias", self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y
