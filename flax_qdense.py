# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Copied code from
# https://github.com/google/flax/blob/master/flax/linen/linear.py and
# https://github.com/google/jax/blob/master/jax/_src/lax/lax.py
# modified to accomodate noise and quantization


from jax._src.lax.lax import (
    standard_primitive,
    _dot_general_batch_rule,
    _dot_general_masking_rule,
    remaining,
    ranges_like,
    transpose,
    DotDimensionNumbers,
    PrecisionLike,
    _canonicalize_precision,
    naryop_dtype_rule,
    _input_dtype,
    _any,
    xops,
    xc,
    _precision_config,
    dtypes,
    xla_client,
)
from typing import (
    Any,
    Optional,
    Callable,
    Sequence,
)
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import masking

from flax.linen.module import Module, compact
from flax.linen.initializers import lecun_normal, zeros

from jax import lax
import jax
import numpy as np
import jax.numpy as jnp


default_kernel_init = lecun_normal()

Array = Any
DType = Any
Dtype = Any  # this could be a real type?
PRNGKey = Any
Shape = Sequence[int]


def _dot_general_shape_rule(
    lhs: Array,
    rhs: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: Optional[DType],
    config: dict,
):
  (lhs_contracting, rhs_contracting), (
      lhs_batch,
      rhs_batch,
  ) = dimension_numbers
  if not all(
      np.all(np.greater_equal(d, 0)) and np.all(np.less(d, lhs.ndim))
      for d in (lhs_contracting, lhs_batch)
  ):
    msg = (
        "dot_general requires lhs dimension numbers to be nonnegative and "
        "less than the number of axes of the lhs value, got "
        f"lhs_batch of {lhs_batch} and lhs_contracting of "
        f"{lhs_contracting} for lhs of rank {lhs.ndim}"
    )
    raise TypeError(msg)
  if not all(
      np.all(np.greater_equal(d, 0)) and np.all(np.less(d, rhs.ndim))
      for d in (rhs_contracting, rhs_batch)
  ):
    msg = (
        "dot_general requires rhs dimension numbers to be nonnegative and "
        "less than the number of axes of the rhs value, got "
        f"rhs_batch of {rhs_batch} and rhs_contracting of "
        f"{rhs_contracting} for rhs of rank {rhs.ndim}"
    )
    raise TypeError(msg)
  if len(lhs_batch) != len(rhs_batch):
    msg = (
        "dot_general requires equal numbers of lhs_batch and rhs_batch "
        "dimensions, got lhs_batch {} and rhs_batch {}."
    )
    raise TypeError(msg.format(lhs_batch, rhs_batch))
  lhs_contracting_set, lhs_batch_set = set(lhs_contracting), set(lhs_batch)
  rhs_contracting_set, rhs_batch_set = set(rhs_contracting), set(rhs_batch)
  if len(lhs_batch_set) != len(lhs_batch):
    msg = (
        "dot_general requires lhs batch dimensions to be distinct, got "
        f"lhs_batch {lhs_batch}."
    )
    raise TypeError(msg)
  if len(rhs_batch_set) != len(rhs_batch):
    msg = (
        "dot_general requires rhs batch dimensions to be distinct, got "
        f"rhs_batch {rhs_batch}."
    )
    raise TypeError(msg)
  if len(lhs_contracting_set) != len(lhs_contracting):
    msg = (
        "dot_general requires lhs contracting dimensions to be distinct, "
        f"got lhs_contracting {lhs_contracting}."
    )
    raise TypeError(msg)
  if len(rhs_contracting_set) != len(rhs_contracting):
    msg = (
        "dot_general requires rhs contracting dimensions to be distinct, "
        f"got rhs_contracting {rhs_contracting}."
    )
    raise TypeError(msg)
  if lhs_contracting_set & lhs_batch_set:
    msg = (
        "dot_general requires lhs batch dimensions to be disjoint from "
        "contracting dimensions, got lhs_batch {} and lhs_contracting {}."
    )
    raise TypeError(msg.format(lhs_batch, lhs_contracting))
  if rhs_contracting_set & rhs_batch_set:
    msg = (
        "dot_general requires rhs batch dimensions to be disjoint from "
        "contracting dimensions, got rhs_batch {} and rhs_contracting {}."
    )
    raise TypeError(msg.format(rhs_batch, rhs_contracting))
  lhs_batch_shape = np.take(lhs.shape, lhs_batch)
  rhs_batch_shape = np.take(rhs.shape, rhs_batch)
  if not np.all(np.equal(lhs_batch_shape, rhs_batch_shape)):
    msg = (
        "dot_general requires lhs batch dimensions and rhs batch "
        "dimensions to have the same shape, got {} and {}."
    )
    raise TypeError(msg.format(lhs_batch_shape, rhs_batch_shape))
  lhs_contracting_shape = np.take(lhs.shape, lhs_contracting)
  rhs_contracting_shape = np.take(rhs.shape, rhs_contracting)
  if not np.all(np.equal(lhs_contracting_shape, rhs_contracting_shape)):
    msg = (
        "dot_general requires contracting dimensions to have the same "
        "shape, got {} and {}."
    )
    raise TypeError(
        msg.format(lhs_contracting_shape, rhs_contracting_shape)
    )

  batch_shape = tuple(lhs_batch_shape)
  lhs_contract_or_batch = tuple(
      sorted(tuple(lhs_contracting) + tuple(lhs_batch))
  )
  lhs_tensored_shape = tuple(np.delete(lhs.shape, lhs_contract_or_batch))
  rhs_contract_or_batch = tuple(
      sorted(tuple(rhs_contracting) + tuple(rhs_batch))
  )
  rhs_tensored_shape = tuple(np.delete(rhs.shape, rhs_contract_or_batch))
  return batch_shape + lhs_tensored_shape + rhs_tensored_shape


def _dot_general_dtype_rule(
    lhs: Array,
    rhs: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: Optional[DType],
    config: dict,
):
  input_dtype = naryop_dtype_rule(
      _input_dtype, [_any, _any], "dot_general", lhs, rhs
  )
  if preferred_element_type is None:
    return input_dtype
  if dtypes.issubdtype(input_dtype, np.integer) and not dtypes.issubdtype(
      preferred_element_type, np.integer
  ):
    raise TypeError(
        "`preferred_element_type` and the original type must both be "
        "integral or both be floating point."
    )
  if dtypes.issubdtype(
      input_dtype, np.signedinteger
  ) and not dtypes.issubdtype(preferred_element_type, np.signedinteger):
    raise TypeError(
        "`preferred_element_type` must have the same signedness as the "
        "original type."
    )
  input_bitwidth = np.dtype(input_dtype).itemsize
  preferred_bitwidth = np.dtype(preferred_element_type).itemsize
  if preferred_bitwidth < input_bitwidth:
    raise TypeError(
        "`preferred_element_type` must not be narrower than the original "
        "type."
    )
  return preferred_element_type


def _dot_general_translation_rule(
    c: Any,
    lhs: Array,
    rhs: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: Optional[DType],
    config: dict,
):
  if preferred_element_type is not None:
    preferred_element_type = xla_client.dtype_to_etype(
        preferred_element_type
    )
  return xops.DotGeneral(
      lhs,
      rhs,
      xc.make_dot_dimension_numbers(dimension_numbers),
      precision_config=_precision_config(precision),
  )


def _dot_general_transpose_lhs(
    g: Array,
    y: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: Optional[DType],
    config: dict,
    swap_ans: bool = False,  # had to keep this one for ad
):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  x_ndim = g.ndim - y.ndim + len(x_batch) + 2 * len(x_contract)
  x_kept = remaining(range(x_ndim), x_contract, x_batch)
  y_kept = remaining(range(y.ndim), y_contract, y_batch)
  if swap_ans:
    ans_batch, ans_y, _ = ranges_like(x_batch, y_kept, x_kept)
  else:
    ans_batch, _, ans_y = ranges_like(x_batch, x_kept, y_kept)
  dims = ((ans_y, y_kept), (ans_batch, y_batch))
  x_contract_sorted_by_y = list(
      np.take(x_contract, np.argsort(y_contract))
  )  # type: ignore[arg-type]
  out_axes = np.argsort(list(x_batch) + x_kept + x_contract_sorted_by_y)

  if config is not None and "err_inpt_noise" in config:
    if config["err_inpt_noise"] != 0.0:
      g = g + jnp.max(g) * config["err_inpt_noise"] * np.random.randn(
          *g.shape
      )

  results = transpose(
      dot_general(
          g,
          y,
          dims,
          precision=precision,
          preferred_element_type=preferred_element_type,
          config=config,
      ),
      tuple(out_axes),
  )

  return results


def _dot_general_transpose_rhs(
    g: Array,
    x: Array,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: Optional[DType],
    config: dict,
):
  (x_contract, y_contract), (x_batch, y_batch) = dimension_numbers
  swapped_dimension_numbers = ((y_contract, x_contract), (y_batch, x_batch))

  if config is not None and "err_weight_noise" in config:
    if config["err_weight_noise"] != 0.0:
      g = g + jnp.max(g) * config["err_weight_noise"] * np.random.randn(
          *g.shape
      )

  results = _dot_general_transpose_lhs(
      g,
      x,
      dimension_numbers=swapped_dimension_numbers,
      precision=precision,
      preferred_element_type=preferred_element_type,
      swap_ans=True,
      config=config,
  )

  return results


dot_general_p = standard_primitive(
    _dot_general_shape_rule,
    _dot_general_dtype_rule,
    "dot_general",
    _dot_general_translation_rule,
)
ad.defbilinear(
    dot_general_p, _dot_general_transpose_lhs, _dot_general_transpose_rhs
)
batching.primitive_batchers[dot_general_p] = _dot_general_batch_rule
masking.masking_rules[dot_general_p] = _dot_general_masking_rule


def dot_general(
    lhs: Array,
    rhs: Array,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike,
    preferred_element_type: Optional[DType],
    config: dict,
) -> Array:
  """More general contraction operator.
  Wraps XLA's `DotGeneral
  <https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`_
  operator.
  Args:
    lhs: an array
    rhs: an array
    dimension_numbers: a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
      (lhs_batch_dims, rhs_batch_dims))`
    precision: Optional. Either ``None``, which means the default precision
      for the backend, a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      ``lax.Precision`` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the
      default accumulation type for the input types, or a datatype,
      indicating to accumulate results to and return a result with that
      datatype.
    config: bit widths and other configurations
  Returns:
    An array containing the result.
  """
  contract_dims_seq, batch_dims_seq = dimension_numbers
  contract_dims = tuple(map(lambda x: tuple(x), contract_dims_seq))
  batch_dims = tuple(map(lambda x: tuple(x), batch_dims_seq))
  return dot_general_p.bind(
      lhs,
      rhs,
      dimension_numbers=(contract_dims, batch_dims),
      precision=_canonicalize_precision(precision),
      preferred_element_type=preferred_element_type,
      config=config,
  )


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

    if self.config is not None and "weight_noise" in self.config:
      if self.config["weight_noise"] != 0.0:
        rng, prng = jax.random.split(rng, 2)
        kernel = kernel + jnp.max(kernel) * self.config[
            "weight_noise"
        ] * jax.random.normal(prng, kernel.shape)

    if self.config is not None and "act_noise" in self.config:
      if self.config["act_noise"] != 0.0:
        rng, prng = jax.random.split(rng, 2)
        inputs = inputs + jnp.max(inputs) * self.config[
            "act_noise"
        ] * jax.random.normal(prng, inputs.shape)

    y = dot_general(
        inputs,
        kernel,
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
        preferred_element_type=None,  # from dot_general default value
        config=self.config,
    )
    if self.use_bias:
      bias = self.param("bias", self.bias_init, (self.features,))
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y
