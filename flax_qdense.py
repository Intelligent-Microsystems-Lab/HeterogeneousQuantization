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
from flax.linen.initializers import lecun_normal  # , zeros

import ml_collections

import jax
import jax.numpy as jnp

from quant import get_noise, signed_uniform_max_scale_quant_ste


default_kernel_init = lecun_normal()

Array = Any
Dtype = Any
PRNGKey = Any
Shape = Sequence[int]


@jax.custom_vjp
def dot_general(inpt: Array, kernel: Array, rng: PRNGKey, cfg: dict) -> Array:
  # Nois
  if "weight_noise" in cfg:
    rng, prng = jax.random.split(rng, 2)
    kernel = kernel + get_noise(kernel, cfg["weight_noise"], prng)

  if "act_noise" in cfg:
    rng, prng = jax.random.split(rng, 2)
    inpt = inpt + get_noise(inpt, cfg["act_noise"], prng)

  # Quantization
  if "weight_bits" in cfg:
    kernel = signed_uniform_max_scale_quant_ste(kernel, cfg["weight_bits"])

  if "act_bits" in cfg:
    inpt = signed_uniform_max_scale_quant_ste(inpt, cfg["act_bits"])

  return jnp.dot(inpt, kernel)


def dot_general_fwd(
    inpt: Array, kernel: Array, rng: PRNGKey, cfg: dict
) -> Array:
  rng, prng = jax.random.split(rng, 2)
  return dot_general(inpt, kernel, rng, cfg,), (
      inpt,
      kernel,
      prng,
      cfg,
  )


def dot_general_bwd(res: tuple, g: Array) -> tuple:
  (
      inpt,
      kernel,
      rng,
      cfg,
  ) = res
  g_inpt = g_weight = g

  # Noise
  if "weight_bwd_noise" in cfg:
    rng, prng = jax.random.split(rng, 2)
    kernel = kernel + get_noise(kernel, cfg["weight_bwd_noise"], prng)

  if "act_bwd_noise" in cfg:
    rng, prng = jax.random.split(rng, 2)
    inpt = inpt + get_noise(inpt, cfg["act_bwd_noise"], prng)

  if "err_inpt_noise" in cfg:
    rng, prng = jax.random.split(rng, 2)
    g_inpt = g_inpt + get_noise(g_inpt, cfg["err_inpt_noise"], prng)

  if "err_weight_noise" in cfg:
    rng, prng = jax.random.split(rng, 2)
    g_weight = g_weight + get_noise(
        g_weight, cfg["err_weight_noise"], prng
    )

  # Quantization
  if "weight_bwd_bits" in cfg:
    kernel = signed_uniform_max_scale_quant_ste(
        kernel, cfg["weight_bwd_bits"]
    )

  if "act_bwd_bits" in cfg:
    inpt = signed_uniform_max_scale_quant_ste(inpt, cfg["act_bwd_bits"])

  if "err_inpt_bits" in cfg:
    g_inpt = signed_uniform_max_scale_quant_ste(
        g_inpt, cfg["err_inpt_bits"]
    )

  if "err_weight_bits" in cfg:
    g_weight = signed_uniform_max_scale_quant_ste(
        g_weight, cfg["err_weight_bits"]
    )

  g_inpt_fwd = jnp.dot(g_inpt, jnp.transpose(kernel))

  g_kernel_fwd = jnp.dot(jnp.transpose(inpt), g_weight)

  return (g_inpt_fwd, g_kernel_fwd, None, None)


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
  # use_bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  # bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  config: dict = ml_collections.FrozenConfigDict({})

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

    y = dot_general(inputs, kernel, rng, dict(self.config))

    # if self.use_bias:
    #   assert False, "Bias for Dense layer not supported yet."
    #   bias = self.param("bias", self.bias_init, (self.features,))
    #   bias = jnp.asarray(bias, self.dtype)
    #   y = y + bias
    return y
