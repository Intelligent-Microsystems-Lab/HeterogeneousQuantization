# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Copied code from
# https://github.com/google/flax/blob/master/flax/linen/linear.py and
# https://github.com/google/jax/blob/master/jax/_src/lax/lax.py
# modified to accomodate noise and quantization

from typing import Any, Callable, Iterable, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax import lax
import jax

from jax._src.lax.lax import (
    Sequence,
    ConvGeneralDilatedDimensionNumbers,
    PrecisionLike,
    conv_dimension_numbers,
    padtype_to_pads,
    _canonicalize_precision,
    ConvDimensionNumbers,
    _conv_sdims,
    _conv_spec_transpose,
    _reshape_axis_out_of,
    _reshape_axis_into,
    _conv_general_vjp_lhs_padding,
    rev,
    _conv_general_vjp_rhs_padding,
    standard_primitive,
    partial,
    xla,
    batching,
    masking,
    ad,
    _conv_general_dilated_masking_rule,
    _conv_general_dilated_batch_rule,
    _conv_general_dilated_dtype_rule,
    _conv_general_dilated_shape_rule,
    _conv_general_dilated_translation_rule,
)

from flax.linen.module import Module, compact
from flax.linen.initializers import zeros
from flax.linen.linear import (
    default_kernel_init,
    PRNGKey,
    Shape,
    Dtype,
    Array,
    _conv_dimension_numbers,
)

#from casting import int_quant, downcast_sat_ftz, get_bounds
#from grad_metrics import metric_grad_single, metric_grad_couple


def conv_general_dilated(
    lhs: Array,
    rhs: Array,
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Optional[Sequence[int]],
    rhs_dilation: Optional[Sequence[int]],
    dimension_numbers: ConvGeneralDilatedDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    precision: PrecisionLike,
    preferred_element_type: Optional[Dtype],
    config: dict,
) -> Array:
  """General n-dimensional convolution operator, with optional dilation.
  Wraps XLA's `Conv
  <https://www.tensorflow.org/xla/operation_semantics#conv_convolution>`_
  operator.
  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply
      before and after each spatial dimension.
    lhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `lhs`. LHS
      dilation is also known as transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. RHS
      dilation is also known as atrous convolution.
    dimension_numbers: either `None`, a `ConvDimensionNumbers` object, or
      a 3-tuple `(lhs_spec, rhs_spec, out_spec)`, where each element is a
      string of length `n+2`.
    feature_group_count: integer, default 1. See XLA HLO docs.
    batch_group_count: integer, default 1. See XLA HLO docs.
    precision: Optional. Either ``None``, which means the default precision
      for the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    config: ?????
  Returns:
    An array containing the convolution result.
  In the string case of `dimension_numbers`, each character identifies by
  position:
  - the batch dimensions in `lhs`, `rhs`, and the output with the character
    'N',
  - the feature dimensions in `lhs` and the output with the character 'C',
  - the input and output feature dimensions in rhs with the characters 'I'
    and 'O' respectively, and
  - spatial dimension correspondences between lhs, rhs, and the output using
    any distinct characters.
  For example, to indicate dimension numbers consistent with the `conv`
  function with two spatial dimensions, one could use `('NCHW', 'OIHW',
  'NCHW')`. As another example, to indicate dimension numbers consistent with
  the TensorFlow Conv2D operation, one could use `('NHWC', 'HWIO', 'NHWC')`.
  When using the latter form of convolution dimension specification, window
  strides are associated with spatial dimension character labels according to
  the order in which the labels appear in the `rhs_spec` string, so that
  `window_strides[0]` is matched with the dimension corresponding to the
  first character appearing in rhs_spec that is not `'I'` or `'O'`.
  If `dimension_numbers` is `None`, the default is `('NCHW', 'OIHW', 'NCHW')`
  (for a 2D convolution).
  """
  dnums = conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  if lhs_dilation is None:
    lhs_dilation = (1,) * (lhs.ndim - 2)
  elif isinstance(padding, str) and not len(
      lhs_dilation
  ) == lhs_dilation.count(1):
    raise ValueError(
        "String padding is not implemented for transposed convolution "
        "using this op. Please either exactly specify the required padding"
        " or use conv_transpose."
    )
  if rhs_dilation is None:
    rhs_dilation = (1,) * (rhs.ndim - 2)
  if isinstance(padding, str):
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = np.take(rhs.shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [
        (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
    ]
    padding = padtype_to_pads(
        np.take(lhs.shape, lhs_perm)[2:],
        effective_rhs_shape,  # type: ignore[index]
        window_strides,
        padding,
    )

  return conv_general_dilated_p.bind(
      lhs,
      rhs,
      window_strides=tuple(window_strides),
      padding=tuple(padding),
      lhs_dilation=tuple(lhs_dilation),
      rhs_dilation=tuple(rhs_dilation),
      dimension_numbers=dnums,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
      lhs_shape=lhs.shape,
      rhs_shape=rhs.shape,
      precision=_canonicalize_precision(precision),
      preferred_element_type=preferred_element_type,
      config=config,
  )


def _conv_general_dilated_transpose_lhs(
    g: Array,
    rhs: Array,
    *,
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Optional[Sequence[int]],
    rhs_dilation: Optional[Sequence[int]],
    dimension_numbers: ConvGeneralDilatedDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    lhs_shape: Tuple,
    rhs_shape: Tuple,
    precision: PrecisionLike,
    preferred_element_type: Optional[Dtype],
    config: dict,
):
  assert type(dimension_numbers) is ConvDimensionNumbers
  assert batch_group_count == 1 or feature_group_count == 1
  lhs_sdims, rhs_sdims, out_sdims = map(_conv_sdims, dimension_numbers)
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  t_rhs_spec = _conv_spec_transpose(rhs_spec)
  if feature_group_count > 1:
    # in addition to switching the dims in the spec, need to move the
    # feature group axis into the transposed rhs's output feature dim
    rhs = _reshape_axis_out_of(rhs_spec[0], feature_group_count, rhs)
    rhs = _reshape_axis_into(rhs_spec[0], rhs_spec[1], rhs)
  elif batch_group_count > 1:
    rhs = _reshape_axis_out_of(rhs_spec[0], batch_group_count, rhs)
    rhs = _reshape_axis_into(rhs_spec[0], rhs_spec[1], rhs)
    feature_group_count = batch_group_count
  trans_dimension_numbers = ConvDimensionNumbers(
      out_spec, t_rhs_spec, lhs_spec
  )
  padding = _conv_general_vjp_lhs_padding(
      np.take(lhs_shape, lhs_sdims),
      np.take(rhs_shape, rhs_sdims),
      window_strides,
      np.take(g.shape, out_sdims),
      padding,
      lhs_dilation,
      rhs_dilation,
  )
  revd_weights = rev(rhs, rhs_sdims)

  if config is not None and 'err_inpt_noise' in config:
    if config['err_inpt_noise'] != 0.:
      g = g + jnp.max(g) * config['err_inpt_noise'] * \
          np.random.randn(*g.shape)

  out = conv_general_dilated(
      g,
      revd_weights,
      window_strides=lhs_dilation,
      padding=padding,
      lhs_dilation=window_strides,
      rhs_dilation=rhs_dilation,
      dimension_numbers=trans_dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=1,
      precision=precision,
      preferred_element_type=preferred_element_type,
      config=config,
  )

  if batch_group_count > 1:
    out = _reshape_axis_out_of(lhs_spec[1], batch_group_count, out)
    out = _reshape_axis_into(lhs_spec[1], lhs_spec[0], out)
  return out


def _conv_general_dilated_transpose_rhs(
    g: Array,
    lhs: Array,
    *,
    window_strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    lhs_dilation: Optional[Sequence[int]],
    rhs_dilation: Optional[Sequence[int]],
    dimension_numbers: ConvDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    lhs_shape: Tuple,
    rhs_shape: Tuple,
    precision: PrecisionLike,
    preferred_element_type: Optional[Dtype],
    config: dict,
):
  assert type(dimension_numbers) is ConvDimensionNumbers
  if np.size(g) == 0:
    # Avoids forming degenerate convolutions where the RHS has spatial size
    # 0. Awkwardly, we don't have an aval for the rhs readily available, so
    # instead of returning an ad_util.Zero instance here, representing a
    # symbolic zero value, we instead return a None, which is meant to
    # represent having no cotangent at all (and is thus incorrect for this
    # situation), since the two are treated the same operationally.
    # TODO(mattjj): adjust defbilinear so that the rhs aval is available
    # here
    return None
  lhs_sdims, rhs_sdims, out_sdims = map(_conv_sdims, dimension_numbers)
  lhs_trans, rhs_trans, out_trans = map(
      _conv_spec_transpose, dimension_numbers
  )
  assert batch_group_count == 1 or feature_group_count == 1
  if batch_group_count > 1:
    feature_group_count = batch_group_count
    batch_group_count = 1
  elif feature_group_count > 1:
    batch_group_count = feature_group_count
    feature_group_count = 1
  trans_dimension_numbers = ConvDimensionNumbers(
      lhs_trans, out_trans, rhs_trans
  )
  padding = _conv_general_vjp_rhs_padding(
      np.take(lhs_shape, lhs_sdims),
      np.take(rhs_shape, rhs_sdims),
      window_strides,
      np.take(g.shape, out_sdims),
      padding,
      lhs_dilation,
      rhs_dilation,
  )

  if config is not None and 'err_weight_noise' in config:
    if config['err_weight_noise'] != 0.:
      g = g + jnp.max(g) * config['err_weight_noise'] * \
          np.random.randn(*g.shape)

  out = conv_general_dilated(
      lhs,
      g,
      window_strides=rhs_dilation,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=window_strides,
      dimension_numbers=trans_dimension_numbers,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
      precision=precision,
      preferred_element_type=preferred_element_type,
      config=config,
  )

  return out


conv_general_dilated_p = standard_primitive(
    _conv_general_dilated_shape_rule,
    _conv_general_dilated_dtype_rule,
    "conv_general_dilated",
    partial(
        _conv_general_dilated_translation_rule,
        expand_complex_convolutions=False,
    ),
)

# TODO(b/161124619, b/161126248): XLA does not support complex convolution on
# CPU or GPU; on these backends, lower complex convolutions away.
xla.backend_specific_translations["cpu"][conv_general_dilated_p] = partial(
    _conv_general_dilated_translation_rule, expand_complex_convolutions=True
)
xla.backend_specific_translations["gpu"][conv_general_dilated_p] = partial(
    _conv_general_dilated_translation_rule, expand_complex_convolutions=True
)

ad.defbilinear(
    conv_general_dilated_p,
    _conv_general_dilated_transpose_lhs,
    _conv_general_dilated_transpose_rhs,
)
batching.primitive_batchers[
    conv_general_dilated_p
] = _conv_general_dilated_batch_rule
masking.masking_rules[
    conv_general_dilated_p
] = _conv_general_dilated_masking_rule


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
  input_dilation: Optional[Iterable[int]] = None
  kernel_dilation: Optional[Iterable[int]] = None
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  config: dict = None

  @compact
  def __call__(self, inputs: Array, rng: Any = None) -> Array:
    """Applies a convolution to the inputs.
    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features)
    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, self.dtype)

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
    kernel = jnp.asarray(kernel, self.dtype)
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    if self.config is not None and 'weight_noise' in self.config:
      if self.config['weight_noise'] != 0.:
        rng, prng = jax.random.split(rng, 2)
        kernel = kernel + \
            jnp.max(kernel) * self.config['weight_noise'] * \
            jax.random.normal(prng, kernel.shape)

    if self.config is not None and 'act_noise' in self.config:
      if self.config['act_noise'] != 0.:
        rng, prng = jax.random.split(rng, 2)
        inputs = inputs + \
            jnp.max(inputs) * self.config['act_noise'] * \
            jax.random.normal(prng, inputs.shape)

    y = conv_general_dilated(
        inputs,
        kernel,
        strides,
        self.padding,
        lhs_dilation=self.input_dilation,
        rhs_dilation=self.kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        batch_group_count=1,  # 1 from conv_general_dilated default values
        precision=self.precision,
        preferred_element_type=None,
        config=self.config,
    )

    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    if self.use_bias:
      # I think i should remove the bias option... maybe?
      bias = self.param("bias", self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y
