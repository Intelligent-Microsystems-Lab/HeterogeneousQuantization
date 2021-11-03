# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax.numpy as jnp


def sse_loss(x, y):
  return jnp.sum((x - y) ** 2)
