import jax
import jax.numpy as jnp

from typing import Any

Array = Any


def signed_uniform_max_scale_quant_ste(x: Array, bits: int) -> Array:
  assert bits > 1, "Bit widths below 2 bits are not supported."

  scale = jnp.max(jnp.abs(x))

  int_range = 2 ** (bits - 1) - 1

  xq = x / scale  # between -1 and 1
  xq = xq * int_range  # scale into valid quant range
  xq = jnp.round(xq)
  xq = xq / int_range
  xq = xq * scale

  return x - jax.lax.stop_gradient(x - xq)
