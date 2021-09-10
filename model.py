# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
import jax.numpy as jnp

from flax import linen as nn


class conv_feature_extractor(nn.Module):
  # DECOLLE based convolutional feature extractor
  # https://arxiv.org/pdf/1811.10766.pdf
  dtype = jnp.bfloat16

  @nn.compact
  def __call__(self, x):

    x = nn.Conv(
        features=64,
        kernel_size=(7, 7),
        dtype=self.dtype,
        padding=[(2, 2), (2, 2)],
    )(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

    x = nn.Conv(
        features=128,
        kernel_size=(7, 7),
        dtype=self.dtype,
        padding=[(2, 2), (2, 2)],
    )(x)
    x = nn.relu(x)

    x = nn.Conv(
        features=128,
        kernel_size=(7, 7),
        dtype=self.dtype,
        padding=[(2, 2), (2, 2)],
    )(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

    return x


def core_fn(params, state, x):
  x = jnp.dot(x, params["w1"]) + jnp.dot(state, params["h1"])
  x = jax.nn.relu(x)

  return x, x


def output_fn(params, state, x):
  x = jnp.dot(x, params["wo"])

  return x, x


def nn_model(params, state, x):
  state, x = core_fn(params["cf"], state, x)
  x, _ = output_fn(params["of"], None, x)

  return state, x


def init_state(out_dim, bs, HIDDEN_SIZE, dtype):
  state = jnp.zeros((bs, HIDDEN_SIZE), dtype=dtype)
  return state


def init_params(rng, inp_dim, out_dim, scale_s, HIDDEN_SIZE):
  rng, w1_rng, h1_rng, wo_rng = jax.random.split(rng, 4)
  params = {
      "cf": {
          "w1": jax.random.normal(w1_rng, (inp_dim, HIDDEN_SIZE))
          * jnp.sqrt(1.0 / inp_dim),
          "h1": jax.random.normal(h1_rng, (HIDDEN_SIZE, HIDDEN_SIZE))
          * jnp.sqrt(1.0 / HIDDEN_SIZE),
      },
      "of": {
          "wo": jax.random.normal(wo_rng, (HIDDEN_SIZE, out_dim))
          * jnp.sqrt(1.0 / HIDDEN_SIZE)
      },
  }

  return params
