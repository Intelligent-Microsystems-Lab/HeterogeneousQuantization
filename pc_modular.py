# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Based on https://github.com/BerenMillidge/PredictiveCodingBackprop

import jax
import jax.numpy as jnp

import numpy as np

from typing import Any, Callable, Iterable, Tuple, Union

from flax import linen as nn
from flax.core import unfreeze, FrozenDict

import ml_collections

from jax._src.lax.lax import (
    conv_dimension_numbers,
    padtype_to_pads,
    ConvDimensionNumbers,
    _conv_sdims,
    _conv_spec_transpose,
    _conv_general_vjp_lhs_padding,
    rev,
    _conv_general_vjp_rhs_padding,
)

from flax.linen.linear import (
    default_kernel_init,
    _conv_dimension_numbers,
)

from quant import get_noise, signed_uniform_max_scale_quant_ste


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any
Array = Any


class ConvolutionalPC(nn.Module):
  features: int
  kernel_size: Iterable[int]
  strides: Union[None, int, Iterable[int]] = None
  padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
  # input_dilation: Union[None, int, Iterable[int]] = 1
  # kernel_dilation: Union[None, int, Iterable[int]] = 1
  # feature_group_count: int = 1
  # use_bias: bool = True
  # dtype: Dtype = jnp.float32
  # precision: Any = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  # bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = zeros
  infer_lr: float = .1
  non_linearity: Callable = jax.nn.relu
  config: dict = None

  @nn.module.compact
  def __call__(self, inputs: Array, rng: PRNGKey):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inputs

    inputs = jnp.asarray(inputs, jnp.float32)
    cfg = self.config

    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    strides = self.strides or (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % 1 == 0
    kernel_shape = kernel_size + (
        in_features // 1,
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

    # Nois
    if "weight_noise" in cfg:
      rng, prng = jax.random.split(rng, 2)
      kernel = kernel + get_noise(kernel, cfg["weight_noise"], prng)

    if "act_noise" in cfg:
      rng, prng = jax.random.split(rng, 2)
      inputs = inputs + get_noise(inputs, cfg["act_noise"], prng)

    # Quantization
    if "weight_bits" in cfg:
      kernel = signed_uniform_max_scale_quant_ste(kernel, cfg["weight_bits"])

    if "act_bits" in cfg:
      inputs = signed_uniform_max_scale_quant_ste(inputs, cfg["act_bits"])

    y = jax.lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=lhs_dilation,
        rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        batch_group_count=1,
        precision=None,
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

    # Noise
    if "weight_bwd_noise" in self.config:
      rng, prng = jax.random.split(rng, 2)
      kernel = kernel + \
          get_noise(kernel, self.config["weight_bwd_noise"], prng)

    if "val_noise" in self.config:
      rng, prng = jax.random.split(rng, 2)
      val = val + get_noise(val, self.config["val_noise"], prng)

    # Quantization
    if "weight_bwd_bits" in self.config:
      kernel = signed_uniform_max_scale_quant_ste(
          kernel, self.config["weight_bwd_bits"]
      )

    if "val_bits" in self.config:
      val = signed_uniform_max_scale_quant_ste(val, self.config["val_bits"])

    pred_err = pred - val
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(
          jax.vmap(jax.vmap(jax.vmap(jax.grad(self.non_linearity))))
      )(out)
      err_prev = deriv * err_prev

    if "err_inpt_noise" in self.config:
      rng, prng = jax.random.split(rng, 2)
      err_prev = err_prev + \
          get_noise(err_prev, self.config["err_inpt_noise"], prng)

    if "err_inpt_bits" in self.config:
      err_prev = signed_uniform_max_scale_quant_ste(
          err_prev, self.config["err_inpt_bits"]
      )

    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size

    if val.ndim == len(kernel_size) + 1:
      val = jnp.expand_dims(val, axis=0)

    strides = self.strides or (1,) * (val.ndim - 2)

    in_features = val.shape[-1]
    assert in_features % 1 == 0
    # kernel_shape = kernel_size + (
    #     in_features // 1,
    #     self.features,
    # )
    # kernel = self.param("kernel", self.kernel_init, kernel_shape)
    kernel = jnp.asarray(kernel, jnp.float32)
    dimension_numbers = _conv_dimension_numbers(val.shape)

    dnums = conv_dimension_numbers(
        val.shape, kernel.shape, dimension_numbers
    )
    rhs_dilation = (1,) * (kernel.ndim - 2)
    lhs_dilation = (1,) * (val.ndim - 2)
    if isinstance(self.padding, str):
      lhs_perm, rhs_perm, _ = dnums
      rhs_shape = np.take(kernel.shape, rhs_perm)[2:]
      effective_rhs_shape = [
          (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
      ]
      padding = padtype_to_pads(
          np.take(val.shape, lhs_perm)[2:],
          effective_rhs_shape,
          strides,
          self.padding,
      )
    else:
      padding = self.padding

    lhs_sdims, rhs_sdims, out_sdims = map(
        _conv_sdims, dimension_numbers
    )
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    t_rhs_spec = _conv_spec_transpose(rhs_spec)
    trans_dimension_numbers = ConvDimensionNumbers(
        out_spec, t_rhs_spec, lhs_spec
    )
    padding_mod = _conv_general_vjp_lhs_padding(
        np.take(val.shape, lhs_sdims),
        np.take(kernel.shape, rhs_sdims),
        strides,
        np.take(err_prev.shape, out_sdims),
        padding,
        lhs_dilation,
        rhs_dilation,
    )
    revd_weights = rev(kernel, rhs_sdims)

    err = jax.lax.conv_general_dilated(
        err_prev,
        revd_weights,
        window_strides=lhs_dilation,
        padding=padding_mod,
        lhs_dilation=strides,
        rhs_dilation=rhs_dilation,
        dimension_numbers=trans_dimension_numbers,
        feature_group_count=1,
        batch_group_count=1,
        precision=None,
        preferred_element_type=None,
    )

    pred -= self.infer_lr * (pred_err - err)
    return err, pred

  def grads(self, err, rng):
    kernel = self.get_variable("params", "kernel")
    val = self.get_variable("pc", "value")

    if "act_bwd_noise" in self.config:
      rng, prng = jax.random.split(rng, 2)
      val = val + get_noise(val, self.config["act_bwd_noise"], prng)

    if "act_bwd_bits" in self.config:
      val = signed_uniform_max_scale_quant_ste(
          val, self.config["act_bwd_bits"])

    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(
          jax.vmap(jax.vmap(jax.vmap(jax.grad(self.non_linearity))))
      )(out)
      err = deriv * err

    if "err_weight_noise" in self.config:
      rng, prng = jax.random.split(rng, 2)
      err = err + get_noise(err, self.config["err_weight_noise"], prng)

    if "err_weight_bits" in self.config:
      err = signed_uniform_max_scale_quant_ste(
          err, self.config["err_weight_bits"]
      )

    if isinstance(self.kernel_size, int):
      kernel_size = (self.kernel_size,)
    else:
      kernel_size = self.kernel_size

    if val.ndim == len(kernel_size) + 1:
      val = jnp.expand_dims(val, axis=0)

    strides = self.strides or (1,) * (val.ndim - 2)

    in_features = val.shape[-1]
    assert in_features % 1 == 0
    # kernel_shape = kernel_size + (
    #     in_features // 1,
    #     self.features,
    # )
    # kernel = self.param("kernel", self.kernel_init, kernel_shape)
    kernel = jnp.asarray(kernel, jnp.float32)
    dimension_numbers = _conv_dimension_numbers(val.shape)

    dnums = conv_dimension_numbers(
        val.shape, kernel.shape, dimension_numbers
    )
    rhs_dilation = (1,) * (kernel.ndim - 2)
    lhs_dilation = (1,) * (val.ndim - 2)
    if isinstance(self.padding, str):
      lhs_perm, rhs_perm, _ = dnums
      rhs_shape = np.take(kernel.shape, rhs_perm)[2:]
      effective_rhs_shape = [
          (k - 1) * r + 1 for k, r in zip(rhs_shape, rhs_dilation)
      ]
      padding = padtype_to_pads(
          np.take(val.shape, lhs_perm)[2:],
          effective_rhs_shape,
          strides,
          self.padding,
      )
    else:
      padding = self.padding

    lhs_sdims, rhs_sdims, out_sdims = map(
        _conv_sdims, dimension_numbers
    )
    lhs_trans, rhs_trans, out_trans = map(
        _conv_spec_transpose, dimension_numbers
    )
    padding_weight = _conv_general_vjp_rhs_padding(
        np.take(val.shape, lhs_sdims),
        np.take(kernel.shape, rhs_sdims),
        strides,
        np.take(err.shape, out_sdims),
        padding,
        lhs_dilation,
        rhs_dilation,
    )
    trans_dimension_numbers = ConvDimensionNumbers(
        lhs_trans, out_trans, rhs_trans
    )

    kernel_grads = jax.lax.conv_general_dilated(
        val,
        err,
        window_strides=rhs_dilation,
        padding=padding_weight,
        lhs_dilation=lhs_dilation,
        rhs_dilation=strides,
        dimension_numbers=trans_dimension_numbers,
        feature_group_count=1,
        batch_group_count=1,
        precision=None,
        preferred_element_type=None,
    )

    return {"kernel": kernel_grads}


class DensePC(nn.Module):
  features: int
  non_linearity: Callable = jax.nn.relu
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  infer_lr: float = .1
  cfg: dict = ml_collections.FrozenConfigDict({})

  @nn.module.compact
  def __call__(self, inpt: Array, rng: PRNGKey):
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inpt

    kernel = self.param(
        "kernel", self.kernel_init, (inpt.shape[-1], self.features)
    )

    # Nois
    if "weight_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      kernel = kernel + get_noise(kernel, self.cfg["weight_noise"], prng)

    if "act_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      inpt = inpt + get_noise(inpt, self.cfg["act_noise"], prng)

    # Quantization
    if "weight_bits" in self.cfg:
      kernel = signed_uniform_max_scale_quant_ste(
          kernel, self.cfg["weight_bits"])

    if "act_bits" in self.cfg:
      inpt = signed_uniform_max_scale_quant_ste(inpt, self.cfg["act_bits"])

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

    # Noise
    if "weight_bwd_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      kernel = kernel + get_noise(kernel, self.cfg["weight_bwd_noise"], prng)

    if "val_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      val = val + get_noise(val, self.cfg["val_noise"], prng)

    # Quantization
    if "weight_bwd_bits" in self.cfg:
      kernel = signed_uniform_max_scale_quant_ste(
          kernel, self.cfg["weight_bwd_bits"]
      )

    if "val_bits" in self.cfg:
      val = signed_uniform_max_scale_quant_ste(val, self.cfg["val_bits"])

    pred_err = pred - val
    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(jax.vmap(jax.grad(self.non_linearity)))(out)
      err_prev = deriv * err_prev

    if "err_inpt_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      err_prev = err_prev + \
          get_noise(err_prev, self.cfg["err_inpt_noise"], prng)

    if "err_inpt_bits" in self.cfg:
      err_prev = signed_uniform_max_scale_quant_ste(
          err_prev, self.cfg["err_inpt_bits"]
      )

    err = jnp.dot(err_prev, jnp.transpose(kernel))
    pred -= self.infer_lr * (pred_err - err)
    return pred_err, pred

  def grads(self, err, rng):
    val = self.get_variable("pc", "value")

    if "act_bwd_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      val = val + get_noise(val, self.cfg["act_bwd_noise"], prng)

    if "act_bwd_bits" in self.cfg:
      val = signed_uniform_max_scale_quant_ste(val, self.cfg["act_bwd_bits"])

    if self.non_linearity is not None:
      out = self.get_variable("pc", "out")
      deriv = jax.vmap(jax.vmap(jax.grad(self.non_linearity)))(out)
      err = deriv * err

    if "err_weight_noise" in self.cfg:
      rng, prng = jax.random.split(rng, 2)
      err = err + get_noise(err, self.cfg["err_weight_noise"], prng)

    if "err_weight_bits" in self.cfg:
      err = signed_uniform_max_scale_quant_ste(
          err, self.cfg["err_weight_bits"]
      )

    return {"kernel": jnp.dot(jnp.transpose(val), err)}


# class MaxPoolPC(nn.Module):
#   window_shape: Iterable[int]
#   strides: Iterable[int] = (1, 1)
#   padding: Union[str, Iterable[Tuple[int, int]]] = "SAME"
#   config: dict = ml_collections.FrozenConfigDict({})

#   @nn.module.compact
#   def __call__(self, inpt: Array, rng: PRNGKey):
#     val = self.variable("pc", "value", jnp.zeros, ())
#     val.value = inpt

#     y = nn.max_pool(inpt, self.window_shape, self.strides, self.padding)

#     return y

#   def infer(self, err_prev, pred, rng):
#     val = self.get_variable("pc", "value")

#     # do some kind of unpooling
#     jax.grad(nn.max_pool)(val, self.window_shape, self.strides, self.padding)

#     return err, _

#   def grads(self, err):
#     return {}


class FlattenPC(nn.Module):
  config: dict = None

  @nn.module.compact
  def __call__(self, inpt: Array, rng: PRNGKey) -> Array:
    val = self.variable("pc", "value", jnp.zeros, ())
    val.value = inpt

    y = inpt.reshape((inpt.shape[0], -1))

    return y

  def infer(self, err_prev: Array, pred: Array, rng: PRNGKey) -> Array:
    val = self.get_variable("pc", "value")

    err = jnp.reshape(err_prev, val.shape)

    return err, jnp.zeros_like(pred)

  def grads(self, err: Array, rng: PRNGKey) -> dict:
    return {}


class PC_NN(nn.Module):
  """Abstract Base PC model."""

  config: dict = None
  loss_fn: Callable = None

  def setup(self):
    self.layers = []

  def __call__(
      self, x: Array, rng: PRNGKey, err_init: bool = False
  ) -> Array:
    err = {}
    for layer in self.layers:
      rng, subkey = jax.random.split(rng, 2)
      x = layer(x, subkey)
      err[layer.name] = jnp.zeros_like(x)

    if err_init:
      return x, err

    return x

  def grads(self, x: Array, y: Array, rng: PRNGKey) -> tuple:
    rng, subkey1, subkey2 = jax.random.split(rng, 3)
    out, err_init = self.__call__(x, subkey1, err_init=True)
    err = self.inference(y, out, subkey2, err_init)

    grads = {}
    for layer in self.layers:
      rng, subkey = jax.random.split(rng, 2)
      g_temp = layer.grads(err[layer.name], subkey)
      if g_temp != {}:
        grads[layer.name] = g_temp

    return FrozenDict(jax.tree_map(lambda x: -1 * x, grads)), out

  def inference(
      self, y: Array, out: Array, rng: PRNGKey, err_init: dict
  ) -> dict:
    pred = unfreeze(self.variables["pc"])
    err_fin = -jax.grad(self.loss_fn)(out, y)
    layer_names = list(pred.keys())

    err_init[layer_names[-1]] = err_fin

    def scan_fn(carry: tuple, _) -> tuple:
      pred, err, rng = carry
      t_err = err_fin
      for layer, l_name in zip(
          self.layers[1:][::-1], layer_names[:-1][::-1]
      ):
        rng, subkey = jax.random.split(rng, 2)
        t_err, pred[layer.name]["value"] = layer.infer(
            t_err, pred[layer.name]["value"], subkey
        )
        err[l_name] = t_err
      return (pred, err, rng), None

    (_, err, _), _ = jax.lax.scan(
        scan_fn,
        (pred, err_init, rng),
        xs=None,
        length=int(self.config.infer_steps),
    )

    return err
