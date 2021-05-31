# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer


import jax
import jax.numpy as jnp


from flax import linen as nn


class conv_feature_extractor(nn.Module):

    dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        # x = x / 16.

        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            dtype=self.dtype,
            padding=[(2, 2), (2, 2)],
        )(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

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
        x = nn.avg_pool(x, window_shape=(3, 3), strides=(3, 3))

        # x = nn.Conv(features=32, kernel_size=(7, 7))(x)
        # x = nn.sigmoid(x)
        # x = nn.max_pool(x, window_shape=(4, 4), strides=(2, 2))

        # x = nn.Conv(features=64, kernel_size=(7, 7))(x)
        # x = nn.sigmoid(x)
        # x = nn.max_pool(x, window_shape=(4, 4), strides=(2, 2))

        # x = nn.Conv(features=128, kernel_size=(7, 7))(x)
        # x = nn.sigmoid(x)

        return x


def core_fn(params, state, x):
    x = jnp.dot(x, params["w1"]) + jnp.dot(state, params["h1"])
    x = jax.nn.relu(x)

    return x, x


def output_fn(params, state, x):
    x = jnp.dot(x, params["wo"])

    return x, x


def nn_model(params, state, x):
    for i in range(len(state)):
        state[i], x = core_fn(params["cf"][i], state[i], x)

    x, _ = output_fn(params["of"], None, x)

    return state, x


def init_state(out_dim, bs, HIDDEN_SIZE, dtype):
    state = []
    for i in HIDDEN_SIZE:
        state.append(jnp.zeros((bs, i), dtype=dtype))
    return state


def init_params(rng, inp_dim, out_dim, scale_s, HIDDEN_SIZE):
    rng, wo_rng = jax.random.split(rng, 2)
    HIDDEN_SIZE = [inp_dim] + HIDDEN_SIZE

    params = {
        "cf": [],
        "of": {
            "wo": jax.random.normal(wo_rng, (HIDDEN_SIZE[-1], out_dim))
            * jnp.sqrt(1.0 / HIDDEN_SIZE[-1])
        },
    }

    for i in range(len(HIDDEN_SIZE) - 1):
        rng, w1_rng, h1_rng = jax.random.split(rng, 3)
        params["cf"].append(
            {
                "w1": jax.random.normal(
                    w1_rng, (HIDDEN_SIZE[i], HIDDEN_SIZE[i + 1])
                )
                * jnp.sqrt(1.0 / HIDDEN_SIZE[i]),
                "h1": jax.random.normal(
                    h1_rng, (HIDDEN_SIZE[i + 1], HIDDEN_SIZE[i + 1])
                )
                * jnp.sqrt(1.0 / HIDDEN_SIZE[i + 1]),
            }
        )

    return params
    # return {
    #     "cf1": {
    #         "w1": jax.random.normal(w1_rng, (inp_dim, HIDDEN_SIZE1))
    #         * jnp.sqrt(1.0 / inp_dim),
    #         "h1": jax.random.normal(h1_rng, (HIDDEN_SIZE1, HIDDEN_SIZE1))
    #         * jnp.sqrt(1.0 / HIDDEN_SIZE1),
    #     },
    #     "cf2": {
    #         "w1": jax.random.normal(w1_rng, (inp_dim, HIDDEN_SIZE2))
    #         * jnp.sqrt(1.0 / inp_dim),
    #         "h1": jax.random.normal(h1_rng, (HIDDEN_SIZE2, HIDDEN_SIZE2))
    #         * jnp.sqrt(1.0 / HIDDEN_SIZE2),
    #     },
    #     "of": {
    #         "wo": jax.random.normal(wo_rng, (HIDDEN_SIZE2, out_dim))
    #         * jnp.sqrt(1.0 / HIDDEN_SIZE2)
    #     },
    # }
