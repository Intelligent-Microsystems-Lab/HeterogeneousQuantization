import jax
import jax.numpy as jnp


HIDDEN_SIZE = 64


# def core_fn(params, state, x):
#     x = jnp.dot(x, params["w1"]) + jnp.dot(state["u1"], params['h1']) + params['b1']
#     x = jax.nn.relu(x)
#     state["u1"] = x

#     return state, x


def core_fn(params, state, x):
    x = jnp.dot(x, params["w1"]) + jnp.dot(state, params["h1"]) + params["b1"]
    x = jax.nn.relu(x)
    state = x

    return state, x


def output_fn(params, x):
    x = jax.nn.relu(jnp.dot(x, params["wo"]) + params["bo"])

    return x


def nn_model(params, state, x):
    state, x = core_fn(params["cf"], state, x)
    x = output_fn(params["of"], x)

    return state, x


def init_state(out_dim, bs):
    return jnp.zeros(
        (
            bs,
            HIDDEN_SIZE,
        )
    )


def init_params(rng, inp_dim, out_dim, scale_s):
    rng, w1_rng, h1_rng, wo_rng = jax.random.split(rng, 4)
    return {
        "cf": {
            "w1": jax.random.normal(
                w1_rng,
                (inp_dim, HIDDEN_SIZE),
            )
            * jnp.sqrt(1.0 / inp_dim),
            "h1": jax.random.normal(
                h1_rng,
                (
                    HIDDEN_SIZE,
                    HIDDEN_SIZE,
                ),
            )
            * jnp.sqrt(1.0 / HIDDEN_SIZE),
            "b1": jnp.zeros((HIDDEN_SIZE,)),
        },
        "of": {
            "wo": jax.random.normal(
                wo_rng,
                (HIDDEN_SIZE, out_dim),
            )
            * jnp.sqrt(1.0 / HIDDEN_SIZE),
            "bo": jnp.zeros((out_dim,)),
        },
    }
