import jax
import jax.numpy as jnp

from typing import Any


Array = Any
PRNGKey = Any


def get_noise(x: Array, percentage: float, rng: PRNGKey) -> Array:
  return (
      jnp.max(jnp.abs(x))
      * percentage
      * jax.random.uniform(rng, x.shape, minval=-1, maxval=1.0)
  )


def signed_uniform_max_scale_quant_ste(x: Array, bits: int) -> Array:
  if type(bits) == int:
    assert (
        bits > 1
    ), "Bit widths below 2 bits are not supported but got bits: " + str(bits)

  scale = jnp.max(jnp.abs(x))

  int_range = 2 ** (bits - 1) - 1

  xq = x / scale  # between -1 and 1
  xq = xq * int_range  # scale into valid quant range
  xq = jnp.round(xq)
  xq = xq / int_range
  xq = xq * scale

  return x - jax.lax.stop_gradient(x - xq)


# parametric homogenouse quantization
class parametric_d(nn.Module):
  # Based on LEARNED STEP SIZE QUANTIZATION
  # https://arxiv.org/abs/1902.08153.
  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass

# parametric heterogenous quantization


class parametric_quant_d_xmax(nn.Module):
  # Based on MIXED PRECISION DNNS
  # https://openreview.net/pdf?id=Hyx0slrFvH
  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass

# differentiable quantization


def differentiable_quant(x: Array, bits: int) -> Array:
  # Based on Differentiable Soft Quantization
  # https://arxiv.org/abs/1908.05033
  pass

# parametric homogenouse differentiable quantization


class parametric_differentiable_d(nn.Module):
  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass

# parametric heterogenous differentiable quantization


class parametric_differentiable_d_xmax(nn.Module):
  @nn.compact
  def __call__(self, inputs: Array,) -> Array:
    pass
