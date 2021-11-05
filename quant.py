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


#
# Rounding with different backward passes
#

# psgd https://arxiv.org/abs/2005.11035 (like)
@jax.custom_vjp
def round_psgd(x, scale):
  return jnp.round(x)


def round_psgd_fwd(x, scale):
  return round_psgd(x, scale), (x, scale)


def round_psgd_bwd(res, g):
  (x, scale) = res

  return (g * (1 + scale * jnp.sign(g) * jnp.abs((x - jnp.round(x)))), None)


round_psgd.defvjp(round_psgd_fwd, round_psgd_bwd)


# ewgs https://arxiv.org/pdf/2104.00903.pdf
@jax.custom_vjp
def round_ewgs(x, scale):
  return jnp.round(x)


def round_ewgs_fwd(x, scale):
  return round_ewgs(x, scale), (x, scale)


def round_ewgs_bwd(res, g):
  (x, scale) = res

  return (g * (1 + scale * jnp.sign(g) * (x - jnp.round(x))), None)


round_ewgs.defvjp(round_ewgs_fwd, round_ewgs_bwd)


@jax.custom_vjp
def round_tanh(x, scale):
  return jnp.round(x)


def round_tanh_fwd(x, scale):
  return round_tanh(x, scale), (x, scale)


def round_tanh_bwd(res, g):
  (x, scale) = res

  # 4 is a parameter to scale the softness/steepness.
  return (g * (1 + scale * jnp.sign(g) * jax.nn.tanh((x - jnp.round(x)
                                                      ) * 4.)), None)


round_tanh.defvjp(round_tanh_fwd, round_tanh_bwd)


@jax.custom_vjp
def round_fsig(x, scale):
  return jnp.round(x)


def round_fsig_fwd(x, scale):
  return round_fsig(x, scale), (x, scale)


def round_fsig_bwd(res, g):
  (x, scale) = res

  # Fast sigmoid derivative
  def fsig_deriv(x):
    return 1 / (1 + jnp.abs(x))**2

  # 2 is a parameter to scale the softness/steepness.
  return (g * (1 + scale * jnp.sign(g) * (fsig_deriv((x + .5 - jnp.round(
      x + .5)) * 2.))), None)


round_fsig.defvjp(round_fsig_fwd, round_fsig_bwd)

# https://arxiv.org/abs/2103.12593
# Copied from https://github.com/byin-cwi/Efficient-spiking-networks/\
# blob/main/DVS128/srnn_class_scnn_enc.ipynb


@jax.custom_vjp
def round_gaussian(x, scale):
  return jnp.round(x)


def round_gaussian_fwd(x, scale):
  return round_gaussian(x, scale), (x, scale)


def round_gaussian_bwd(res, g):
  (x, scale) = res

  lens = .5

  def gaussian_deriv(x):
    return jnp.exp(-(x**2) / (2 * lens**2)) / jnp.sqrt(2 * jnp.pi) / lens

  return (g * (1 + scale * jnp.sign(g) * gaussian_deriv((x + .5 - jnp.round(
      x + .5)) * 3)), None)


round_gaussian.defvjp(round_gaussian_fwd, round_gaussian_bwd)

# https://arxiv.org/abs/2103.12593
# Copied from https://github.com/byin-cwi/Efficient-spiking-networks/\
# blob/main/DVS128/srnn_class_scnn_enc.ipynb


@jax.custom_vjp
def round_multi_gaussian(x, scale):
  return jnp.round(x)


def round_multi_gaussian_fwd(x, scale):
  return round_multi_gaussian(x, scale), (x, scale)


def round_multi_gaussian_bwd(res, g):
  (x, scale) = res

  # Fast sigmoid derivative
  lens = .5
  hight = .15
  scale_gaussian = 6.0

  def gaussian_fn(x, mu, sigma):
    return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / jnp.sqrt(
        2 * jnp.pi) / sigma

  def multi_gaussian_deriv(x):
    return gaussian_fn(x, mu=0., sigma=lens) * (
        1. + hight) - gaussian_fn(
        x, mu=lens, sigma=scale_gaussian * lens) * hight - gaussian_fn(
        x, mu=- lens, sigma=scale_gaussian * lens) * hight

  return (g * (1 + scale * jnp.sign(g) * multi_gaussian_deriv((
      x + .5 - jnp.round(x + .5)) * 3)), None)


round_multi_gaussian.defvjp(round_multi_gaussian_fwd, round_multi_gaussian_bwd)


#
# Calibration functions
#


def max_init(x, bits, sign):
  return jnp.max(jnp.abs(x))


def double_mean_init(x, bits, sign):
  return 2 * jnp.mean(jnp.abs(x))


def gaussian_init(x, bits, sign):
  mu = jnp.mean(x)
  sigma = jnp.std(x)
  return jnp.maximum(jnp.abs(mu - 3 * sigma), jnp.abs(mu + 3 * sigma))


def percentile_init(x, bits, sign, perc):
  return jnp.percentile(jnp.abs(x), perc)


#
# Quantizer
#


class uniform_static(nn.Module):
  bits: int = 8
  act: bool = False
  round_fn: Callable = round_psgd
  init_fn: Callable = max_init
  g_scale: float = 0.

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
      xmax.value = self.init_fn(x, bits=self.bits, sign=sign)
      xmax.value = jnp.where(xmax.value == 0, 1., xmax.value)

    if sign:
      xmin = -xmax.value
    else:
      xmin = 0.

    scale = xmax.value / num_levels
    return self.round_fn(jnp.clip(x, xmin, xmax.value) / scale,
                         self.g_scale) * scale


class parametric_d(nn.Module):
  bits: int = 8
  act: bool = False
  round_fn: Callable = round_psgd
  init_fn: Callable = max_init
  g_scale: float = 0.

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
      step_size.value *= self.init_fn(inputs,
                                      bits=self.bits,
                                      sign=sign) / jnp.sqrt(q_pos)

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
    vbar = self.round_fn(v, self.g_scale)
    return vbar * s


class parametric_d_xmax(nn.Module):
  bits: int = 8  # here its just init bits
  act: bool = False
  xmax_min: float = 0.001
  xmax_max: float = 100
  d_min: float = 2**-8
  d_max: float = 2**+8
  round_fn: Callable = round_psgd
  init_fn: Callable = max_init
  g_scale: float = 0.

  # Parametric heterogenous quantization.
  # Based on MIXED PRECISION DNNS.
  # https://openreview.net/pdf?id=Hyx0slrFvH
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:

    x = inputs

    def quantize_pow2(v):
      return 2 ** round_psgd(jnp.log2(v), 0)

    @jax.custom_vjp
    def ceilpass(x):
      return jnp.ceil(x)

    def ceilpass_fwd(x):
      return ceilpass(x), (None,)

    def ceilpass_bwd(res, g):
      return (g,)

    ceilpass.defvjp(ceilpass_fwd, ceilpass_bwd)

    if sign:
      num_levels = 2 ** (self.bits - 1) - 1
    else:
      num_levels = 2 ** (self.bits) - 1

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    d = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    xmax = self.variable(
        'quant_params', 'dynamic_range', jnp.ones, (1,))

    act_mb = self.variable('act_size', 'act_mb', jnp.ones, (1,))
    weight_mb = self.variable('weight_size', 'weight_mb', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      xmax.value = jnp.clip(self.init_fn(inputs, bits=self.bits, sign=sign),
                            self.xmax_min, self.xmax_max)
      xmax.value = jnp.where(xmax.value == 0, 1., xmax.value)
      d.value = jnp.clip(xmax.value / num_levels, self.d_min, self.d_max)

    # Ensure that stepsize is in specified range (and a power of two).
    d = jnp.clip(d.value, self.d_min, self.d_max)
    # Ensure that dynamic range is in specified range.
    xmax = jnp.clip(xmax.value, self.xmax_min, self.xmax_max)

    # Ensure xmax and d do not exceed each other
    d = jnp.where(d > xmax, xmax, d)
    xmax = jnp.where(xmax < d, d, xmax)

    # Aux scope to compute network size on the fly.
    if self.is_mutable_collection('act_size'):
      if self.act:
        n_wf = inputs.shape[1:]
        if sign:
          act_mb.value = np.prod(
              n_wf) * (ceilpass(jnp.log2((xmax / d) + 1)) + 1) / 8_000_000
        else:
          act_mb.value = np.prod(
              n_wf) * (ceilpass(jnp.log2((xmax / d) + 1))) / 8_000_000
      else:
        act_mb.value = 0.

    if self.is_mutable_collection('weight_size'):
      if self.act:
        weight_mb.value = 0.
      else:
        n_wf = inputs.shape
        if sign:
          weight_mb.value = np.prod(
              n_wf) * (ceilpass(jnp.log2((xmax / d) + 1)) + 1) / 8_000_000
        else:
          weight_mb.value = np.prod(
              n_wf) * (ceilpass(jnp.log2((xmax / d) + 1))) / 8_000_000

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

      g_d = (1 / d) * (quant(x, d, xmax) - x)

      if sign:
        g_xmax = jnp.where(x > xmax, 1, 0)
        g_xmax = jnp.where(x < xmin, -1, g_xmax)
      else:
        g_xmax = jnp.where(x > xmax, 1, 0)

      return g * mask, jnp.sum(g * g_d * mask), jnp.sum(g * g_xmax)
    quant.defvjp(quant_fwd, quant_bwd)

    return quant(x, d, xmax)  # quantize_pow2
