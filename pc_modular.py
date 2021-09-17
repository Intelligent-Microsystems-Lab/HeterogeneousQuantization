# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Based on https://github.com/BerenMillidge/PredictiveCodingBackprop

import jax
import jax.numpy as jnp

import numpy as np

from typing import Any, Callable, Iterable, Tuple, Union

from flax import linen as nn
from flax.core import unfreeze, FrozenDict
from flax.linen.initializers import lecun_normal, zeros

from jax._src.lax.lax import (
    _canonicalize_precision,
    conv_dimension_numbers,
    padtype_to_pads,
    ConvDimensionNumbers,
    _conv_sdims,
    _conv_spec_transpose,
    _conv_general_vjp_lhs_padding,
    _conv_general_vjp_rhs_padding,
    _reshape_axis_out_of,
    _reshape_axis_into,
    rev,
    conv_general_dilated,
)

default_kernel_init = lecun_normal()


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


def get_noise(x, percentage, rng):
  return (
      jnp.max(jnp.abs(x))
      * percentage
      * jax.random.uniform(rng, x.shape, minval=-1, maxval=1.0)
  )


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def RecurrentPC():
  pass


def conv_fwd(
    lhs,
    rhs,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers,
    feature_group_count,
    precision,
    batch_group_count,
    preferred_element_type,
):
  dnums = conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
  if lhs_dilation is None:
    lhs_dilation = (1,) * (lhs.ndim - 2)
  elif isinstance(padding, str) and not len(
      lhs_dilation
  ) == lhs_dilation.count(1):
    raise ValueError(
        "String padding is not implemented for transposed convolution "
        "using this op. Please either exactly specify the required padding or "
        "use conv_transpose."
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

  return jax.lax.conv_general_dilated(
      lhs,
      rhs,
      window_strides=tuple(window_strides),
      padding=tuple(padding),
      lhs_dilation=tuple(lhs_dilation),
      rhs_dilation=tuple(rhs_dilation),
      dimension_numbers=dnums,
      feature_group_count=feature_group_count,
      batch_group_count=batch_group_count,
      # lhs_shape=lhs.shape, rhs_shape=rhs.shape,
      precision=_canonicalize_precision(precision),
      preferred_element_type=preferred_element_type,
  )


def conv_bwd_inpt(
    g,
    rhs,
    *,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers,
    feature_group_count,
    batch_group_count,
    lhs_shape,
    rhs_shape,
    precision,
    preferred_element_type,
):
  dnums = conv_dimension_numbers(lhs_shape, rhs_shape, dimension_numbers)
  if isinstance(padding, str):
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = np.take(rhs_shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [
        (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
    ]
    padding = padtype_to_pads(
        np.take(lhs_shape, lhs_perm)[2:],
        effective_rhs_shape,  # type: ignore[index]
        window_strides,
        padding,
    )

  assert type(dimension_numbers) is ConvDimensionNumbers
  assert batch_group_count == 1 or feature_group_count == 1
  lhs_sdims, rhs_sdims, out_sdims = map(_conv_sdims, dimension_numbers)
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  t_rhs_spec = _conv_spec_transpose(rhs_spec)
  if feature_group_count > 1:
    # in addition to switching the dims in the spec, need to move the feature
    # group axis into the transposed rhs's output feature dim
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
  )
  if batch_group_count > 1:
    out = _reshape_axis_out_of(lhs_spec[1], batch_group_count, out)
    out = _reshape_axis_into(lhs_spec[1], lhs_spec[0], out)
  return out


def conv_bwd_kernel(
    g,
    lhs,
    *,
    window_strides,
    padding,
    lhs_dilation,
    rhs_dilation,
    dimension_numbers: ConvDimensionNumbers,
    feature_group_count: int,
    batch_group_count: int,
    lhs_shape,
    rhs_shape,
    precision,
    preferred_element_type,
):

  dnums = conv_dimension_numbers(lhs_shape, rhs_shape, dimension_numbers)
  if isinstance(padding, str):
    lhs_perm, rhs_perm, _ = dnums
    rhs_shape = np.take(rhs_shape, rhs_perm)[2:]  # type: ignore[index]
    effective_rhs_shape = [
        (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
    ]
    padding = padtype_to_pads(
        np.take(lhs_shape, lhs_perm)[2:],
        effective_rhs_shape,  # type: ignore[index]
        window_strides,
        padding,
    )

  assert type(dimension_numbers) is ConvDimensionNumbers
  if np.size(g) == 0:
    # Avoids forming degenerate convolutions where the RHS has spatial size 0.
    # Awkwardly, we don't have an aval for the rhs readily available, so
    # instead of returning an ad_util.Zero instance here, representing a
    # symbolic zero value, we instead return a None, which is meant to
    # represent having no cotangent at all (and is thus incorrect for this
    # situation), since the two are treated the same operationally.
    # TODO(mattjj): adjust defbilinear so that the rhs aval is available here
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
  return conv_general_dilated(
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
  )


class ConvolutionalPC(nn.Module):
  features: int
  kernel_size: Iterable[int]
  strides: Union[None, int, Iterable[int]] = 1
  padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
  input_dilation: Union[None, int, Iterable[int]] = 1
  kernel_dilation: Union[None, int, Iterable[int]] = 1
  feature_group_count: int = 1
  use_bias: bool = True
  dtype: Dtype = jnp.float32
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  non_linearity: Callable = jax.nn.relu
  config: dict = None

  @nn.module.compact
  def __call__(self, inputs, rng):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inputs

    subkey1, subkey2 = jax.random.split(rng, 2)

    # flax conv prep
    inputs = jnp.asarray(inputs, self.dtype)
    if isinstance(self.kernel_size, int):
      raise TypeError(
          "The kernel size must be specified as a"
          " tuple/list of integers (eg.: [3, 3])."
      )
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return x

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)
    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)
    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    kernel_shape = kernel_size + (
        in_features // self.feature_group_count,
        self.features,
    )
    kernel = self.param("kernel", self.kernel_init, kernel_shape)
    kernel = jnp.asarray(kernel, self.dtype)
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    if self.config is not None and "weight_noise" in self.config:
      kernel = kernel + jnp.max(jnp.abs(kernel)) * self.config[
          "weight_noise"
      ] * jax.random.uniform(subkey1, kernel.shape)

    if self.config is not None and "act_noise" in self.config:
      inputs = inputs + jnp.max(jnp.abs(inputs)) * self.config[
          "act_noise"
      ] * jax.random.uniform(subkey2, inputs.shape)

    y = conv_fwd(
        lhs=inputs,
        rhs=kernel,
        window_strides=strides,
        padding=self.padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision,
        batch_group_count=1,
        preferred_element_type=None,
    )

    if is_single_input:
      y = jnp.squeeze(y, axis=0)

    if self.non_linearity is not None:
      out = self.variable("pc", "out", jnp.zeros, ())
      out.value = y
      y = self.non_linearity(y)

    return y

  def infer(self, err_prev, pred, rng):
    val = self.get_variable("pc", "value")
    kernel = self.get_variable("params", "kernel")

    subkey1, subkey2, subkey3 = jax.random.split(rng, 3)
    if self.config is not None and "weight_noise" in self.config:
      kernel = kernel + jnp.max(jnp.abs(kernel)) * self.config[
          "weight_noise"
      ] * jax.random.uniform(subkey1, kernel.shape)

    if self.config is not None and "act_noise" in self.config:
      val = val + jnp.max(jnp.abs(val)) * self.config[
          "act_noise"
      ] * jax.random.uniform(subkey2, val.shape)

    pred_err = pred - val
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(
          jax.vmap(jax.vmap(jax.vmap(jax.grad(self.non_linearity))))
      )(out)
      err_prev = deriv * err_prev

    # flax conv prep
    inputs = jnp.asarray(val, self.dtype)
    if isinstance(self.kernel_size, int):
      raise TypeError(
          "The kernel size must be specified as a"
          " tuple/list of integers (eg.: [3, 3])."
      )
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return x

    if inputs.ndim == len(kernel_size) + 1:
      inputs = jnp.expand_dims(inputs, axis=0)
    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)
    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    if self.config is not None and "err_inpt_noise" in self.config:
      err_prev = err_prev + jnp.max(jnp.abs(err_prev)) * self.config[
          "err_inpt_noise"
      ] * jax.random.uniform(subkey3, err_prev.shape)

    err = conv_bwd_inpt(
        g=err_prev,
        rhs=kernel,
        window_strides=strides,
        padding=self.padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        lhs_shape=inputs.shape,
        rhs_shape=kernel.shape,
        precision=self.precision,
        batch_group_count=1,
        preferred_element_type=None,
    )

    pred -= self.config.infer_lr * (pred_err - err)
    return err, pred

  def grads(self, err, rng):
    kernel = self.get_variable("params", "kernel")
    val = self.get_variable("pc", "value")

    subkey1, subkey2 = jax.random.split(rng, 2)
    if self.config is not None and "act_noise" in self.config:
      val = val + jnp.max(jnp.abs(val)) * self.config[
          "act_noise"
      ] * jax.random.uniform(subkey1, val.shape)

    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(
          jax.vmap(jax.vmap(jax.vmap(jax.grad(self.non_linearity))))
      )(out)
      err = deriv * err

    # flax conv prep
    inputs = jnp.asarray(val, self.dtype)
    if isinstance(self.kernel_size, int):
      raise TypeError(
          "The kernel size must be specified as a"
          " tuple/list of integers (eg.: [3, 3])."
      )
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return x

    if inputs.ndim == len(kernel_size) + 1:
      inputs = jnp.expand_dims(inputs, axis=0)
    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)
    in_features = inputs.shape[-1]
    assert in_features % self.feature_group_count == 0
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    if self.config is not None and "err_weight_noise" in self.config:
      err = err + jnp.max(jnp.abs(err)) * self.config[
          "err_weight_noise"
      ] * jax.random.uniform(subkey2, err.shape)

    kernel_grads = conv_bwd_kernel(
        g=err,
        lhs=val,
        window_strides=strides,
        padding=self.padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        lhs_shape=inputs.shape,
        rhs_shape=kernel.shape,
        precision=self.precision,
        batch_group_count=1,
        preferred_element_type=None,
    )
    return {"kernel": kernel_grads}


class DensePC(nn.Module):
  features: int
  non_linearity: Callable = jax.nn.relu
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  config: dict = None

  @nn.module.compact
  def __call__(self, inpt, rng):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inpt

    kernel = self.param(
        "kernel", self.kernel_init, (inpt.shape[-1], self.features)
    )

    subkey1, subkey2 = jax.random.split(rng, 2)
    if self.config is not None and "weight_noise" in self.config:
      kernel = kernel + get_noise(kernel, self.config["weight_noise"], subkey1)

    if self.config is not None and "act_noise" in self.config:
      inpt = inpt + get_noise(inpt, self.config["act_noise"], subkey2)

    y = jax.lax.dot_general(
        inpt,
        kernel,
        (((inpt.ndim - 1,), (0,)), ((), ())),
    )

    if self.non_linearity is not None:
      out = self.variable("pc", "out", jnp.zeros, ())
      out.value = y
      y = self.non_linearity(y)

    return y

  def infer(self, err_prev, pred, rng):
    val = self.get_variable("pc", "value")
    kernel = self.get_variable("params", "kernel")

    subkey1, subkey2, subkey3 = jax.random.split(rng, 3)

    if self.config is not None and "weight_noise" in self.config:
      kernel = kernel + get_noise(kernel, self.config["weight_noise"], subkey1)

    if self.config is not None and "val_noise" in self.config:
      val = val + get_noise(val, self.config["val_noise"], subkey2)

    pred_err = pred - val
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(jax.vmap(jax.grad(self.non_linearity)))(out)
      err_prev = deriv * err_prev

    if self.config is not None and "err_inpt_noise" in self.config:
      err_prev = err_prev + \
          get_noise(err_prev, self.config["err_inpt_noise"], subkey3)

    err = jnp.dot(err_prev, jnp.transpose(kernel))
    pred -= self.config.infer_lr * (pred_err - err)
    return pred_err, pred

  def grads(self, err, rng):
    val = self.get_variable("pc", "value")

    subkey1, subkey2 = jax.random.split(rng, 2)

    if self.config is not None and "act_noise" in self.config:
      val = val + get_noise(val, self.config["act_noise"], subkey1)

    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(jax.vmap(jax.grad(self.non_linearity)))(out)
      err = deriv * err

    if self.config is not None and "err_weight_noise" in self.config:
      err = err + get_noise(err, self.config["err_weight_noise"], subkey2)

    return {"kernel": jnp.dot(jnp.transpose(val), err)}


# class MaxPoolPC(nn.Module):
#   window_shape
#   strides=None
#   padding="VALID"
#   config: dict = None

#   @nn.module.compact
#   def __call__(self, inpt):
#     val = self.variable("pc", "value", jnp.zeros, ())
#     val.value = inpt

#     y = nn.max_pool(inpt, window_shape, strides, padding)

#     return y

#   def infer(self, err_prev, pred):
#     val = self.get_variable("pc", "value")

#     # do some kind of unpooling
#     err = jnp.dot(err_prev, jnp.transpose(kernel))

#     return err, _

#   def grads(self, err):
#     return {}


class FlattenPC(nn.Module):
  config: dict = None

  @nn.module.compact
  def __call__(self, inpt, rng):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inpt

    y = inpt.reshape((inpt.shape[0], -1))

    return y

  def infer(self, err_prev, pred, rng):
    val = self.get_variable("pc", "value")

    err = jnp.reshape(err_prev, val.shape)

    return err, jnp.zeros_like(pred)

  def grads(self, err, rng):
    return {}


class PC_NN(nn.Module):
  """Abstract Base PC model."""

  config: dict = None
  loss_fn: Callable = None

  def setup(self):
    self.layers = []

  def __call__(self, x, rng, err_init=False):
    err = {}
    for l in self.layers:
      rng, subkey = jax.random.split(rng, 2)
      x = l(x, subkey)
      err[l.name] = jnp.zeros_like(x)

    if err_init:
      return x, err

    return x

  def grads(self, x, y, rng):
    rng, subkey1, subkey2 = jax.random.split(rng, 3)
    out, err_init = self.__call__(x, subkey1, err_init=True)
    err = self.inference(y, out, subkey2, err_init)

    grads = {}
    for l in self.layers:
      rng, subkey = jax.random.split(rng, 2)
      g_temp = l.grads(err[l.name], subkey)
      if g_temp != {}:
        grads[l.name] = g_temp

    return FrozenDict(jax.tree_map(lambda x: -1 * x, grads)), out

  def inference(self, y, out, rng, err_init):
    pred = unfreeze(self.variables["pc"])
    err_fin = -jax.grad(self.loss_fn)(out, y)
    layer_names = list(pred.keys())

    err_init[layer_names[-1]] = err_fin

    def scan_fn(carry, _):
      pred, err, rng = carry
      t_err = err_fin
      for l, l_name in zip(
          self.layers[1:][::-1], layer_names[:-1][::-1]
      ):
        rng, subkey = jax.random.split(rng, 2)
        t_err, pred[l.name]["value"] = l.infer(
            t_err, pred[l.name]["value"], subkey
        )
        err[l_name] = t_err
      return (pred, err, rng), None

    (_, err, _), _ = jax.lax.scan(
        scan_fn,
        (pred, err_init, rng),
        xs=None,
        length=self.config.infer_steps,
    )

    return err
