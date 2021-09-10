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

default_kernel_init = lecun_normal()


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


def MaxPoolPC():
  pass


def ConvolutionalPC():
  pass


def RecurrentPC():
  pass


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
    return err, pred

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
