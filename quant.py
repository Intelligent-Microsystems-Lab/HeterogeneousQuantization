import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, Iterable, Callable


Array = Any
PRNGKey = Any
Shape = Iterable[int]


def get_noise(x: Array, percentage: float, rng: PRNGKey) -> Array:
  return (
      jnp.max(jnp.abs(x))
      * percentage
      * jax.random.uniform(rng, x.shape, minval=-1, maxval=1.0)
  )


@jax.custom_vjp
def roundpass(x):
  return jnp.round(x)


def roundpass_fwd(x):
  return roundpass(x), (None,)


def roundpass_bwd(res, g):
  return (g,)


roundpass.defvjp(roundpass_fwd, roundpass_bwd)


@jax.custom_vjp
def roundsurrogate(x):
  return jnp.round(x)


def roundsurrogate_fwd(x):
  return roundsurrogate(x), (x,)


def roundsurrogate_bwd(res, g):
  (x,) = res
  diff = jnp.round(x) - x
  g_x = 1 / (1 + jnp.abs(diff))**2
  return (g * g_x,)


roundsurrogate.defvjp(roundsurrogate_fwd, roundsurrogate_bwd)


# surrogate with psg
@jax.custom_vjp
def round_surrogate_psg(x):
  return jnp.round(x)


def round_surrogate_psg_fwd(x):
  return round_surrogate_psg(x), (x,)


def round_surrogate_psg_bwd(res, g):
  (x,) = res
  diff = jnp.abs(jnp.round(x) - x) - .5
  g_x = 1 / (1 + jnp.abs(diff))**2

  return (g * g_x,)


round_surrogate_psg.defvjp(round_surrogate_psg_fwd, round_surrogate_psg_bwd)


# ewgs https://arxiv.org/pdf/2104.00903.pdf
@jax.custom_vjp
def round_ewgs(x):
  return jnp.round(x)


def round_ewgs_fwd(x):
  return round_ewgs(x), (x,)


def round_ewgs_bwd(res, g):
  (x,) = res

  return (g * (1 + 1e-3 * jnp.sign(g) * (x - jnp.round(x))),)


round_ewgs.defvjp(round_ewgs_fwd, round_ewgs_bwd)


@jax.custom_vjp
def ceilpass(x):
  return jnp.ceil(x)


def ceilpass_fwd(x):
  return ceilpass(x), (None,)


def ceilpass_bwd(res, g):
  return (g,)


ceilpass.defvjp(ceilpass_fwd, ceilpass_bwd)


def max_init(x):
  return jnp.max(jnp.abs(x))


def double_mean_init(x):
  return 2 * jnp.mean(jnp.abs(x))


def gaussian_init(x):
  mu = jnp.mean(x)
  sigma = jnp.std(x)
  return jnp.maximum(jnp.abs(mu - 3 * sigma), jnp.abs(mu + 3 * sigma))


class uniform_dynamic(nn.Module):
  bits: int = 8
  round_fn: Callable = roundpass
  init_fn: Callable = max_init

  @nn.compact
  def __call__(self, x: Array, sign: bool = True) -> Array:
    if type(self.bits) == int:
      assert (
          self.bits > 1
      ), "Bit widths below 2 bits are not supported but got bits: "\
          + str(self.bits)

    if sign:
      scale = self.init_fn(x)
      int_range = 2 ** (self.bits - 1) - 1
    else:
      scale = self.init_fn(x)
      int_range = 2 ** (self.bits) - 1

    xq = x / scale  # between -1 and 1
    xq = xq * int_range  # scale into valid quant range
    xq = self.round_fn(xq)
    xq = xq / int_range
    xq = xq * scale

    if sign:
      jnp.clip(xq, 0, scale)

    return xq


class uniform_static(nn.Module):
  bits: int = 8
  round_fn: Callable = roundpass
  init_fn: Callable = max_init

  @nn.compact
  def __call__(self, x: Array, sign: bool = True) -> Array:
    if type(self.bits) == int:
      assert (
          self.bits > 1
      ), "Bit widths below 2 bits are not supported but got bits: "\
          + str(self.bits)

    if sign:
      num_levels = 2 ** (self.bits - 1) - 1
    else:
      num_levels = 2 ** (self.bits) - 1

    xmax = self.variable('quant_params', 'dynamic_range', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      xmax.value = self.init_fn(x)

    if sign:
      xmin = -xmax.value
    else:
      xmin = 0.

    scale = xmax.value / num_levels
    return self.round_fn(jnp.clip(x, xmin, xmax.value) / scale) * scale


class parametric_d(nn.Module):
  bits: int = 8
  act: bool = False
  round_fn: Callable = roundpass
  init_fn: Callable = double_mean_init

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
      step_size.value *= self.init_fn(inputs) / jnp.sqrt(q_pos)

    gradScaleFactor = 1 / jnp.sqrt(q_pos * np.prod(n_wf) + 1e-6)

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
    vbar = self.round_fn(v)
    return vbar * s


# def total_weight_mb(state):
#   layer_sizes = jax.tree_util.tree_multimap(lambda x, y: jnp.prod(x.shape)
# * to_bits(
#       y['step_size'], y['dynamic_range']), state.params['params'],
# state.params['quant_params'])
#   return jnp.sum(jax.tree_util.flatten(layer_size)[0])


# def total_act_mb(state):
#   pass


class parametric_d_xmax(nn.Module):
  bits: int = 8  # here its just init bits
  act: bool = False
  xmax_min: float = 0.001
  xmax_max: float = 100
  d_min: float = 2**-8
  d_max: float = 2**+8
  round_fn: Callable = roundpass
  init_fn: Callable = max_init

  # Parametric heterogenous quantization.
  # Based on MIXED PRECISION DNNS.
  # https://openreview.net/pdf?id=Hyx0slrFvH
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    x = inputs

    #
    # DISCLAIMER: not using quantize_pow2.
    #
    def quantize_pow2(v):
      return 2 ** roundpass(jnp.log2(v))

    if sign:
      num_levels = 2 ** (self.bits - 1) - 1
    else:
      num_levels = 2 ** (self.bits) - 1

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    d = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    xmax = self.variable(
        'quant_params', 'dynamic_range', jnp.ones, (1,))

    act_mb = self.variable('aux', 'act_mb', jnp.ones, (1,))
    weight_mb = self.variable('aux', 'weight_mb', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      xmax.value = jnp.clip(self.init_fn(inputs),
                            self.xmax_min, self.xmax_max)
      # xmax.value = jnp.clip(2 * jnp.mean(jnp.abs(inputs)),
      #                       self.xmax_min, self.xmax_max)
      d.value = jnp.clip(xmax.value / num_levels, self.d_min, self.d_max)
    # Aux scope to compute network size on the fly.
    if self.is_mutable_collection('aux'):
      if self.act:
        n_wf = inputs.shape[1:]
        act_mb.value = np.prod(
            n_wf) * (ceilpass(jnp.log2(xmax.value / d.value + 1)) + 1) / 8000
        weight_mb = 0.
      else:
        n_wf = inputs.shape
        weight_mb.value = np.prod(
            n_wf) * (ceilpass(jnp.log2(xmax.value / d.value + 1)) + 1) / 8000
        act_mb = 0.

    @jax.custom_vjp
    def quant(x, d, xmax):
      if sign:
        xmin = -xmax
      else:
        xmin = 0.

      return d * jnp.round(jnp.clip(x, xmin, xmax) / d)

    def quant_fwd(x, d, xmax):
      return quant(x, d, xmax), (x, d, xmax)

    def quant_bwd(res, g):
      (x, d, xmax) = res

      if sign:
        xmin = -xmax
      else:
        xmin = 0.

      mask = jnp.where(jnp.logical_and((x < xmax), (x > xmin)), 1, 0)

      g_d = 1 / d * (quant(x, d, xmax) - x)

      if self.round_fn.__name__ == 'roundsurrogate':
        aux_x = jnp.clip(x, xmin, xmax) / d
        diff = jnp.round(aux_x) - aux_x
        g *= 1 / (1 + jnp.abs(diff))**2

      if sign:
        g_xmax = jnp.where(x > xmax, 1, 0)
        g_xmax = jnp.where(x < xmin, -1, g_xmax)
      else:
        g_xmax = jnp.where(x > xmax, 1, 0)

      return g * mask, jnp.sum(g * g_d * mask), jnp.sum(g * g_xmax)
    quant.defvjp(quant_fwd, quant_bwd)

    # Ensure that stepsize is in specified range and a power of two.
    d = jnp.clip(d.value, self.d_min, self.d_max)
    # Ensure that dynamic range is in specified range.
    xmax = jnp.clip(xmax.value, self.xmax_min, self.xmax_max)

    return quant(x, d, xmax)
