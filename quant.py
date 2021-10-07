import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from typing import Any, Callable, Iterable


Array = Any
PRNGKey = Any
Shape = Iterable[int]


def lsq_init(q_pos: int, data: Array) -> Callable:
  """
  Initializes step size for LSQ.
  """
  def init(key: PRNGKey, shape: Shape) -> float:
    step_size = jnp.ones(shape, dtype=jnp.float32)

    return step_size * 2 * jnp.max(jnp.abs(data)) / jnp.sqrt(q_pos)
  return init


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
      scale = jnp.max(jnp.abs(x))
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


class parametric_d(nn.Module):
  bits: int = 8

  # parametric homogenouse quantization
  # Based on LEARNED STEP SIZE QUANTIZATION
  # https://arxiv.org/abs/1902.08153.
  @nn.compact
  def __call__(self, inputs: Array, sign: bool = True) -> Array:
    if sign:
      q_pos = 2 ** (self.bits - 1) - 1
      q_neg = -q_pos
    else:
      q_pos = 2 ** (self.bits) - 1
      q_neg = 0

    # Intialize step size. Step size only changes when init is called or apply
    # with mutable = ['quant_params'].
    step_size = self.variable('quant_params', 'step_size', jnp.ones, (1,))
    if self.is_mutable_collection('quant_params'):
      step_size.value *= 2 * jnp.max(jnp.abs(inputs)) / jnp.sqrt(q_pos)

    @jax.custom_vjp
    def lsq(x, s):
      return jnp.round(jnp.clip(x / s, q_neg, q_pos)) * s

    def lsq_fwd(x, s):
      return lsq(x, s), (x, s)

    def lsq_bwd(res, g):
      x, s = res

      sg = -x / s + jnp.round(x / s)
      sg = jnp.where(x / s <= q_neg, q_neg, sg)
      sg = jnp.where(x / s >= q_pos, q_pos, sg)

      g_scale = 1 / jnp.sqrt(q_pos * np.prod(x.shape))

      gx_mask = jnp.where(x / s <= q_neg, 0., jnp.ones_like(g))
      gx_mask = jnp.where(x / s >= q_pos, 0., gx_mask)

      return gx_mask * g, jnp.sum(sg * g) * g_scale

    lsq.defvjp(lsq_fwd, lsq_bwd)

    return lsq(inputs, step_size.value)


class parametric_quant_d_xmax(nn.Module):
  sign: bool = True

  # parametric heterogenous quantization
  # Based on MIXED PRECISION DNNS
  # https://openreview.net/pdf?id=Hyx0slrFvH
  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass


def differentiable_quant(x: Array, bits: int, sign: bool = True) -> Array:
  # differentiable quantization
  # Based on Differentiable Soft Quantization
  # https://arxiv.org/abs/1908.05033
  pass

# parametric homogenouse differentiable quantization


class parametric_differentiable_d(nn.Module):
  sign: bool = True

  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass

# parametric heterogenous differentiable quantization


class parametric_differentiable_d_xmax(nn.Module):
  sign: bool = True

  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass
