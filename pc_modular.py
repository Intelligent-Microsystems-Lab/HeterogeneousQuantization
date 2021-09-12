# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Based on https://github.com/BerenMillidge/PredictiveCodingBackprop

import jax
import jax.numpy as jnp

import numpy as np

from typing import Any, Callable, Iterable, Optional, Tuple, Union

from flax import linen as nn
from flax.core import freeze, unfreeze, FrozenDict
from flax.linen.initializers import lecun_normal, variance_scaling, zeros

from jax._src.lax.lax import _conv_general_dilated_transpose_lhs, _conv_general_dilated_transpose_rhs, _canonicalize_precision, conv_dimension_numbers, padtype_to_pads

default_kernel_init = lecun_normal()


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def MaxPoolPC():
  pass


def RecurrentPC():
  pass


class ConvolutionalPC(nn.Module):
  features: int
  kernel_size: Iterable[int]
  strides: Union[None, int, Iterable[int]] = 1
  padding: Union[str, Iterable[Tuple[int, int]]] = 'SAME'
  input_dilation: Union[None, int, Iterable[int]] = 1
  kernel_dilation: Union[None, int, Iterable[int]] = 1
  feature_group_count: int = 1
  non_linearity: Callable = jax.nn.relu
  precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  config: dict = None

  def maybe_broadcast(self, x):
    if x is None:
      # backward compatibility with using None as sentinel for
      # broadcast 1
      x = 1
    if isinstance(x, int):
      return (x,) * len(self.kernel_size)
    return x

  @nn.module.compact
  def __call__(self, inpt):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inpt

    in_features = inpt.shape[-1]
    assert in_features % self.feature_group_count == 0
    strides = self.maybe_broadcast(self.strides)
    input_dilation = self.maybe_broadcast(self.input_dilation)
    kernel_dilation = self.maybe_broadcast(self.kernel_dilation)
    kernel_shape = self.kernel_size + (
        in_features // self.feature_group_count, self.features)
    kernel = self.param("kernel", self.kernel_init, kernel_shape)

    dimension_numbers = _conv_dimension_numbers(inpt.shape)
    y = jax.lax.conv_general_dilated(
        inpt,
        kernel,
        strides,
        self.padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        precision=self.precision)

    if self.non_linearity is not None:
      out = self.variable("pc", "out", jnp.zeros, ())
      out.value = y
      y = self.non_linearity(y)

    return y

  def infer(self, err_prev, pred):
    val = self.get_variable("pc", "value")
    kernel = self.get_variable("params", "kernel")

    strides = self.maybe_broadcast(self.strides)
    input_dilation = self.maybe_broadcast(self.input_dilation)
    kernel_dilation = self.maybe_broadcast(self.kernel_dilation)
    dimension_numbers = _conv_dimension_numbers(val.shape)

    pred_err = pred - val
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.jacfwd(self.non_linearity)(out)
      err_prev = jnp.einsum("xzyuabcd,abcd->abcd", deriv, err_prev)


    if isinstance(self.padding, str):
      rhs_dilation = (1,) * (kernel.ndim - 2)
      dnums = conv_dimension_numbers(val.shape, kernel.shape, dimension_numbers)
      lhs_perm, rhs_perm, _ = dnums
      rhs_shape = np.take(kernel.shape, rhs_perm)[2:]  # type: ignore[index]
      effective_rhs_shape = [(k-1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)]
      padding = padtype_to_pads(
          np.take(val.shape, lhs_perm)[2:], effective_rhs_shape,  # type: ignore[index]
          strides, self.padding)

    err = _conv_general_dilated_transpose_lhs(
        g=err_prev,
        rhs=kernel,
        window_strides=strides,
        padding=padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        batch_group_count=1,
        lhs_shape=val.shape,
        rhs_shape=kernel.shape,
        precision=_canonicalize_precision(self.precision),
        preferred_element_type=None)

    pred -= self.config.infer_lr * (pred_err - err)
    return err, pred

  def grads(self, err):
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)
    dimension_numbers = _conv_dimension_numbers(inputs.shape)

    val = self.get_variable("pc", "value")
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.jacfwd(self.non_linearity)(out)
      err = jnp.einsum("xzyuabcd,abcd->abcd", deriv, err)

    kernel_grads = _conv_general_dilated_transpose_rhs(
        g=err,
        lhs=val,
        window_strides=strides,
        padding=self.padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=self.feature_group_count,
        batch_group_count=1,
        lhs_shape=val.shape,
        rhs_shape=kernel.shape,
        precision=_canonicalize_precision(self.precision),
        preferred_element_type=None)
    return {"kernel": kernel_grads}


class DensePC(nn.Module):
  features: int
  non_linearity: Callable = jax.nn.relu
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  config: dict = None

  @nn.module.compact
  def __call__(self, inpt):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inpt

    kernel = self.param(
        "kernel", self.kernel_init, (inpt.shape[-1], self.features)
    )

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

  def infer(self, err_prev, pred):
    val = self.get_variable("pc", "value")
    kernel = self.get_variable("params", "kernel")

    pred_err = pred - val
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.jacfwd(self.non_linearity)(out)
      err_prev = jnp.einsum("dcab,ab->ab", deriv, err_prev)

    err = jnp.dot(err_prev, jnp.transpose(kernel))
    pred -= self.config.infer_lr * (pred_err - err)
    return pred_err, pred

  def grads(self, err):
    val = self.get_variable("pc", "value")
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.jacfwd(self.non_linearity)(out)
      err = jnp.einsum("dcab,ab->ab", deriv, err)
    return {"kernel": jnp.dot(jnp.transpose(val), err)}


class PC_NN(nn.Module):
  """A simple PC model."""

  config: dict = None
  loss_fn: Callable = None

  def setup(self):
    self.layers = []

  def __call__(self, x):
    for l in self.layers:
      x = l(x)
    return x

  def grads(self, x, y):
    out = self.__call__(x)
    err = self.inference(y, out)

    grads = {}
    for l in self.layers:
      grads[l.name] = l.grads(err[l.name])
    return FrozenDict(grads)

  def inference(self, y, out):
    pred = unfreeze(self.variables["pc"])
    err_fin = -jax.grad(self.loss_fn)(out, y)
    layer_names = list(pred.keys())

    err_init = {}
    for i in layer_names:
      err_init[i] = self.variables["pc"][i]["out"]
    err_init[layer_names[-1]] = err_fin

    def scan_fn(carry, _):
      pred, err = carry
      t_err = err_fin
      for l, l_name in zip(
          reversed(self.layers[1:]), reversed(layer_names[:1])
      ):
        t_err, pred[l.name]["value"] = l.infer(
            t_err, pred[l.name]["value"]
        )
        err[l_name] = t_err
      return (pred, err), _

    (_, err), _ = jax.lax.scan(
        scan_fn, (pred, err_init), xs=None, length=self.config.infer_steps
    )

    return err
