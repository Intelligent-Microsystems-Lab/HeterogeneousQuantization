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

import ml_collections

import jax
import jax.numpy as jnp

from quant import get_noise


default_kernel_init = lecun_normal()

Array = Any
Dtype = Any
PRNGKey = Any
Shape = Sequence[int]


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
  config: dict = ml_collections.FrozenConfigDict({})
  bits: int = 8
  quant_act_sign: bool = True
  g_scale: float = 1e-3

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

    # Quantization has to be done here to use Flax convenience functions for
    # parameters.
    if "weight" in self.config:
      kernel_fwd = self.config.weight(
          bits=self.bits, g_scale=self.g_scale)(kernel)
    else:
      kernel_fwd = kernel

    if "act" in self.config:
      inpt_fwd = self.config.act(bits=self.bits, g_scale=self.g_scale)(
          inputs, sign=self.quant_act_sign)
    else:
      inpt_fwd = inputs

    @jax.custom_vjp
    def dot_general(inpt_fwd: Array, kernel_fwd: Array, inpt_bwd: Array,
                    kernel_bwd: Array, rng: PRNGKey) -> Array:
      # Nois
      if "weight_noise" in self.config:
        rng, prng = jax.random.split(rng, 2)
        kernel_fwd = kernel_fwd + \
            get_noise(kernel_fwd, self.config["weight_noise"], prng)

      if "act_noise" in self.config:
        rng, prng = jax.random.split(rng, 2)
        inpt_fwd = inpt_fwd + \
            get_noise(inpt_fwd, self.config["act_noise"], prng)

      return jnp.dot(inpt_fwd, kernel_fwd)

    def dot_general_fwd(
        inpt_fwd: Array, kernel_fwd: Array, inpt_bwd: Array, kernel_bwd: Array,
        rng: PRNGKey
    ) -> Array:
      if rng is not None:
        rng, prng = jax.random.split(rng, 2)
      else:
        prng = None
      return dot_general(inpt_fwd, kernel_fwd, inpt_bwd, kernel_bwd, rng), (
          inpt_bwd,
          kernel_bwd,
          prng,
      )

    def dot_general_bwd(res: tuple, g: Array) -> tuple:
      (
          inpt,
          kernel,
          rng,
      ) = res
      g_inpt = g_weight = g

      # Noise
      if "weight_bwd_noise" in self.config:
        rng, prng = jax.random.split(rng, 2)
        kernel = kernel + \
            get_noise(kernel, self.config["weight_bwd_noise"], prng)

      if "act_bwd_noise" in self.config:
        rng, prng = jax.random.split(rng, 2)
        inpt = inpt + get_noise(inpt, self.config["act_bwd_noise"], prng)

      if "err_inpt_noise" in self.config:
        rng, prng = jax.random.split(rng, 2)
        g_inpt = g_inpt + \
            get_noise(g_inpt, self.config["err_inpt_noise"], prng)

      if "err_weight_noise" in self.config:
        rng, prng = jax.random.split(rng, 2)
        g_weight = g_weight + get_noise(
            g_weight, self.config["err_weight_noise"], prng
        )

      # Quantization
      if "weight_bwd" in self.config:
        kernel = self.config.weight_bwd()(kernel)

      if "act_bwd" in self.config:
        inpt = self.config.act_bwd()(inpt, sign=self.quant_act_sign)

      if "err_inpt" in self.config:
        g_inpt = self.config.err_inpt()(g_inpt)

      if "err_weight" in self.config:
        g_weight = self.config.err_weight()(g_weight)

      g_inpt_fwd = jnp.dot(g_inpt, jnp.transpose(kernel))

      g_kernel_fwd = jnp.dot(jnp.transpose(inpt), g_weight)

      return (g_inpt_fwd, g_kernel_fwd, None, None, None)

    dot_general.defvjp(dot_general_fwd, dot_general_bwd)

    y = dot_general(inpt_fwd, kernel_fwd, inputs, kernel, rng)

    if self.use_bias:
      # assert False, "Bias for Dense layer not supported yet."
      bias = self.param("bias", self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)

      if "bias" in self.config:
        bias = self.config.weight(bits=self.bits, g_scale=self.g_scale)(bias)

      y = y + bias
    return y
