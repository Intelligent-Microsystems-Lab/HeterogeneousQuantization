import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, Iterable


Array = Any
PRNGKey = Any
Shape = Iterable[int]


def get_noise(x: Array, percentage: float, rng: PRNGKey) -> Array:
  return (
      jnp.max(jnp.abs(x))
      * percentage
      * jax.random.uniform(rng, x.shape, minval=-1, maxval=1.0)
  )


class signed_uniform_max_scale_quant_ste(nn.Module):
  bits: int = 8

  @nn.compact
  def __call__(self, x: Array, sign: bool = True) -> Array:
    if type(self.bits) == int:
      assert (
          self.bits > 1
      ), "Bit widths below 2 bits are not supported but got bits: "\
          + str(self.bits)

    if sign:
      scale = jnp.max(jnp.abs(x)) + jnp.finfo(x.dtype).eps
      int_range = 2 ** (self.bits - 1) - 1
    else:
      scale = jnp.max(x)
      int_range = 2 ** (self.bits) - 1

    xq = x / scale  # between -1 and 1
    xq = xq * int_range  # scale into valid quant range
    xq = jnp.round(xq)
    xq = xq / int_range
    xq = xq * scale

    if sign:
      jnp.clip(xq, 0, scale)

    return x - jax.lax.stop_gradient(x - xq)


@jax.custom_vjp
def roundpass(x):
  return jnp.round(x)


def roundpass_fwd(x):
  return roundpass(x), (None,)


def roundpass_bwd(res, g):
  return (g,)


roundpass.defvjp(roundpass_fwd, roundpass_bwd)


class parametric_d(nn.Module):
  bits: int = 8
  act: bool = False

  # parametric homogenouse quantization
  # Based on LEARNED STEP SIZE QUANTIZATION
  # https://arxiv.org/abs/1902.08153.
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    v = inputs

    if sign:
      q_pos = 2 ** (self.bits - 1) - 1
      q_neg = -q_pos

    else:
      q_pos = 2 ** (self.bits) - 1
      q_neg = 0

    if self.act:
      n_wf = v.shape[1:]
    else:
      n_wf = v.shape

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    step_size = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      step_size.value = jnp.ones((1,))
      step_size.value *= 2 * jnp.mean(jnp.abs(inputs)) / jnp.sqrt(q_pos)

    gradScaleFactor = 1 / jnp.sqrt(q_pos * np.prod(n_wf))

    @jax.custom_vjp
    def gradscale(x, scale):
      return x

    def gradscale_fwd(x, scale):
      return gradscale(x, scale), (scale,)

    def gradscale_bwd(res, g):
      (scale,) = res
      return g * scale, None
    gradscale.defvjp(gradscale_fwd, gradscale_bwd)

    s = gradscale(step_size.value, gradScaleFactor)
    v = v / s
    v = jnp.clip(v, q_neg, q_pos)
    vbar = roundpass(v)
    return vbar * s


# def total_weight_mb(state):
#   layer_sizes = jax.tree_util.tree_multimap(lambda x, y: jnp.prod(x.shape)
# * to_bits(
#       y['step_size'], y['dynamic_range']), state.params['params'],
# state.params['quant_params'])
#   return jnp.sum(jax.tree_util.flatten(layer_size)[0])


# def total_act_mb(state):
#   pass


class parametric_quant_d_xmax(nn.Module):
  bits: int = 8  # here its just init bits
  act: bool = False
  xmax_min: float = 0.001
  xmax_max: float = 10
  d_min: float = 2**-8
  d_max: float = 2**+8

  # parametric heterogenous quantization
  # Based on MIXED PRECISION DNNS
  # https://openreview.net/pdf?id=Hyx0slrFvH
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    x = inputs

    def quantize_pow2(v):
      return 2 ** jnp.round(jnp.log2(v))

    if sign:
      num_levels = 2 ** (self.bits - 1) - 1
    else:
      num_levels = 2 ** (self.bits) - 1

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    d = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    xmax = self.variable(
        'quant_params', 'dynamic_range', jnp.ones, (1,))

    size_mb = self.variable('aux', 'size_mb', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      xmax.value = jnp.max(jnp.abs(inputs)) * .8
      d.value = xmax.value / num_levels
    # Aux scope to compute network size on the fly.
    if self.is_mutable_collection('aux'):
      if self.act:
        n_wf = inputs.shape[1:]
      else:
        n_wf = inputs.shape

      size_mb.value = np.prod(
          n_wf) * (jnp.ceil(jnp.log2(xmax.value / d.value + 1)) + 1) / 8000

    # ensure that stepsize is in specified range and a power of two
    d = quantize_pow2(jnp.clip(d, self.d_min, self.d_max))
    # ensure that dynamic range is in specified range
    xmax = jnp.clip(xmax, self.xmax_min, self.xmax_max)

    if sign:
      xmin = -xmax
    else:
      xmin = 0.

    return d * roundpass(jnp.clip(x, xmin, xmax) / d)


# def differentiable_quant(x: Array, bits: int, sign: bool = True) -> Array:
#   # differentiable quantization
#   # Based on Differentiable Soft Quantization
#   # https://arxiv.org/abs/1908.05033
#   pass

# # parametric homogenouse differentiable quantization


# class parametric_differentiable_d(nn.Module):
#   sign: bool = True

#   @nn.compact
#   def __call__(self, inputs: Array,) -> Array:
#     pass

# # parametric heterogenous differentiable quantization


# class parametric_differentiable_d_xmax(nn.Module):
#   sign: bool = True

#   @nn.compact
#   def __call__(self, inputs: Array,) -> Array:
#     pass
