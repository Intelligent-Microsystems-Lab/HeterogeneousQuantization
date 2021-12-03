# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Copied code from
# https://github.com/google/flax/blob/master/flax/linen/linear.py and
# https://github.com/google/jax/blob/master/jax/_src/lax/lax.py
# modified to accomodate noise and quantization

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import jax

from jax._src.lax.lax import (
    conv_dimension_numbers,
    padtype_to_pads,
    ConvDimensionNumbers,
    _conv_sdims,
    _conv_spec_transpose,
    _conv_general_vjp_lhs_padding,
    rev,
    _conv_general_vjp_rhs_padding,
    _reshape_axis_out_of,
    _reshape_axis_into,
)

from flax.linen.module import Module, compact
from flax.linen.linear import (
    default_kernel_init,
    PRNGKey,
    Shape,
    Dtype,
    Array,
    _conv_dimension_numbers,
)

from quant import get_noise


class QuantConv(Module):
  """Convolution Module wrapping lax.conv_general_dilated.
  Attributes:
    features: number of convolution filters.
    kernel_size: shape of the convolutional kernel. For 1D convolution,
      the kernel size can be passed as an integer. For all other cases, it
      must be a sequence of integers.
    strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply
      before and after each spatial dimension.
    input_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `inputs`.
      Convolution with input dilation `d` is equivalent to transposed
      convolution with stride `d`.
    kernel_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of the convolution
      kernel. Convolution with kernel dilation is also known as 'atrous
      convolution'.
    feature_group_count: integer, default 1. If specified divides the input
      features into groups.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer for the convolutional kernel.
    bias_init: initializer for the bias.
    config: ???
  """

  features: int
  kernel_size: Union[int, Iterable[int]]
  strides: Optional[Iterable[int]] = None
  padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
  # input_dilation: Optional[Iterable[int]] = None
  # kernel_dilation: Optional[Iterable[int]] = None
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  # precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  # bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  config: dict = None
  bits: int = 8
  quant_act_sign: bool = True
  g_scale: float = 0.

  @compact
  def __call__(self, inputs: Array, rng: Any = None) -> Array:
    """Applies a convolution to the inputs.
    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features)
    Returns:
      The convolved data.
    """
    assert self.use_bias is False
    inputs = jnp.asarray(inputs, jnp.float32)
    cfg = self.config

    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size  # type: ignore

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    strides = self.strides or (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = kernel_size + (
        in_features // self.feature_group_count,
        self.features,
    )
    kernel = self.param("kernel", self.kernel_init, kernel_shape)
    kernel = jnp.asarray(kernel, jnp.float32)
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    dnums = conv_dimension_numbers(
        inputs.shape, kernel.shape, dimension_numbers
    )
    rhs_dilation = (1,) * (kernel.ndim - 2)
    lhs_dilation = (1,) * (inputs.ndim - 2)
    if isinstance(self.padding, str):
      lhs_perm, rhs_perm, _ = dnums
      rhs_shape = np.take(kernel.shape, rhs_perm)[2:]
      effective_rhs_shape = [
          (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
      ]
      padding = padtype_to_pads(
          np.take(inputs.shape, lhs_perm)[2:],
          effective_rhs_shape,
          strides,
          self.padding,
      )
    else:
      padding = self.padding

    # Quantization has to be done here to use Flax convenience functions for
    # parameters.
    if "weight" in cfg:
      kernel_fwd = cfg.weight(bits=self.bits, g_scale=self.g_scale)(kernel)
    else:
      kernel_fwd = kernel

    if "act" in cfg:
      inpt_fwd = cfg.act(bits=self.bits, g_scale=self.g_scale)(inputs, sign=self.quant_act_sign)
    else:
      inpt_fwd = inputs

    @jax.custom_vjp
    def conv_general(inpt_fwd: Array, kernel_fwd: Array, inpt: Array,
                     kernel: Array, rng: PRNGKey) -> Array:

      # Nois
      if "weight_noise" in cfg:
        rng, prng = jax.random.split(rng, 2)
        kernel_fwd = kernel_fwd + \
            get_noise(kernel_fwd, cfg["weight_noise"], prng)

      if "act_noise" in cfg:
        rng, prng = jax.random.split(rng, 2)
        inpt_fwd = inpt_fwd + get_noise(inpt_fwd, cfg["act_noise"], prng)

      # # Quantization
      # if "weight" in cfg:
      #   kernel = cfg.weight(bits=self.bits)(kernel)

      # if "act" in cfg:
      #   inpt = cfg.act(bits=self.bits)(inpt, sign=self.quant_act_sign)

      return jax.lax.conv_general_dilated(
          inpt_fwd,
          kernel_fwd,
          strides,
          padding,
          lhs_dilation=None,
          rhs_dilation=rhs_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=self.feature_group_count,
          batch_group_count=1,
          precision=None,
          preferred_element_type=None,
      )

    def conv_general_fwd(
        inpt_fwd: Array, kernel_fwd: Array, inpt_bwd: Array,
        kernel_bwd: Array, rng: PRNGKey
    ) -> Array:
      if rng is not None:
        rng, prng = jax.random.split(rng, 2)
      else:
        prng = None
      return conv_general(inpt_fwd, kernel_fwd, inpt_bwd, kernel_bwd,
                          prng), (inpt_bwd, kernel_bwd, rng)

    def conv_general_bwd(res: tuple, g: Array) -> tuple:
      (inpt, kernel, rng) = res
      g_inpt = g_weight = g

      # Noise
      if "weight_bwd_noise" in cfg:
        rng, prng = jax.random.split(rng, 2)
        kernel = kernel + get_noise(
            kernel, cfg["weight_bwd_noise"], prng
        )

      if "act_bwd_noise" in cfg:
        rng, prng = jax.random.split(rng, 2)
        inpt = inpt + get_noise(inpt, cfg["act_bwd_noise"], prng)

      if "err_inpt_noise" in cfg:
        rng, prng = jax.random.split(rng, 2)
        g_inpt = g_inpt + get_noise(
            g_inpt, cfg["err_inpt_noise"], prng
        )

      if "err_weight_noise" in cfg:
        rng, prng = jax.random.split(rng, 2)
        g_weight = g_weight + get_noise(
            g_weight, cfg["err_weight_noise"], prng
        )

      # Quantization
      if "weight_bwd" in cfg:
        kernel = cfg.weight_bwd()(kernel)

      if "act_bwd" in cfg:
        inpt = cfg.act_bwd()(inpt, sign=self.quant_act_sign)

      if "err_inpt" in cfg:
        g_inpt = cfg.err_inpt()(g_inpt)

      if "err_weight" in cfg:
        g_weight = cfg.err_weight()(g_weight)

      lhs_sdims, rhs_sdims, out_sdims = map(
          _conv_sdims, dimension_numbers
      )
      lhs_trans, rhs_trans, out_trans = map(
          _conv_spec_transpose, dimension_numbers
      )
      padding_weight = _conv_general_vjp_rhs_padding(
          np.take(inpt.shape, lhs_sdims),
          np.take(kernel.shape, rhs_sdims),
          strides,
          np.take(g.shape, out_sdims),
          padding,
          lhs_dilation,
          rhs_dilation,
      )
      if self.feature_group_count > 1:
        rhs_conv_batch_group_count = self.feature_group_count
        rhs_conv_feature_group_count = 1
      else:
        rhs_conv_batch_group_count = 1
        rhs_conv_feature_group_count = 1
      trans_dimension_numbers = ConvDimensionNumbers(
          lhs_trans, out_trans, rhs_trans
      )

      g_kernel_fwd = jax.lax.conv_general_dilated(
          inpt,
          g_weight,
          window_strides=rhs_dilation,
          padding=padding_weight,
          lhs_dilation=lhs_dilation,
          rhs_dilation=strides,
          dimension_numbers=trans_dimension_numbers,
          feature_group_count=rhs_conv_feature_group_count,
          batch_group_count=rhs_conv_batch_group_count,
          precision=None,
          preferred_element_type=None,
      )

      lhs_sdims, rhs_sdims, out_sdims = map(
          _conv_sdims, dimension_numbers
      )
      lhs_spec, rhs_spec, out_spec = dimension_numbers
      t_rhs_spec = _conv_spec_transpose(rhs_spec)

      if self.feature_group_count > 1:
        # in addition to switching the dims in the spec, need to move the
        # feature group axis into the transposed rhs's output feature dim
        kernel = _reshape_axis_out_of(
            rhs_spec[0], self.feature_group_count, kernel)
        kernel = _reshape_axis_into(rhs_spec[0], rhs_spec[1], kernel)

      trans_dimension_numbers = ConvDimensionNumbers(
          out_spec, t_rhs_spec, lhs_spec
      )
      padding_mod = _conv_general_vjp_lhs_padding(
          np.take(inputs.shape, lhs_sdims),
          np.take(kernel.shape, rhs_sdims),
          strides,
          np.take(g.shape, out_sdims),
          padding,
          lhs_dilation,
          rhs_dilation,
      )
      revd_weights = rev(kernel, rhs_sdims)

      g_inpt_fwd = jax.lax.conv_general_dilated(
          g_inpt,
          revd_weights,
          window_strides=lhs_dilation,
          padding=padding_mod,
          lhs_dilation=strides,
          rhs_dilation=rhs_dilation,
          dimension_numbers=trans_dimension_numbers,
          feature_group_count=self.feature_group_count,
          batch_group_count=1,
          precision=None,
          preferred_element_type=None,
      )
      return (g_inpt_fwd, g_kernel_fwd, None, None, None)

    conv_general.defvjp(conv_general_fwd, conv_general_bwd)

    y = conv_general(
        inpt_fwd,
        kernel_fwd,
        inputs,
        kernel,
        rng,
    )

    if is_single_input:
      y = jnp.squeeze(y, axis=0)

    # if self.use_bias:
    #   bias = self.param("bias", self.bias_init, (self.features,))
    #   bias = jnp.asarray(bias, self.dtype)
    #   y = y + bias
    return y
