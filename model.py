import jax
import jax.numpy as jnp


HIDDEN1_SIZE = 2048
HIDDEN2_SIZE = 1024


def nn_model(params, state, x):
    # two layer feedforward statefull NN

    x = jnp.dot(x, params["w1"]) + params["h1"] * state["u1"] + params["b1"]
    x = jax.nn.relu(x)
    state["u1"] = x

    x = jnp.dot(x, params["w2"]) + params["h2"] * state["u2"] + params["b2"]
    x = jax.nn.relu(x)
    state["u2"] = x

    x = jnp.dot(x, params["w3"]) + params["h3"] * state["u3"] + params["b3"]
    x = jax.nn.relu(x)
    state["u3"] = x

    return state, x


def init_state(out_dim, bs):
    return {
        "u1": jnp.zeros(
            (
                bs,
                HIDDEN1_SIZE,
            )
        ),
        "u2": jnp.zeros(
            (
                bs,
                HIDDEN2_SIZE,
            )
        ),
        "u3": jnp.zeros(
            (
                bs,
                out_dim,
            )
        ),
    }


def init_params(rng, inp_dim, out_dim, scale_s):
    rng, w1_rng, w2_rng, w3_rng, t1_rng, t2_rng, t3_rng = jax.random.split(
        rng, 7
    )
    return {
        "w1": jax.random.normal(
            w1_rng,
            (inp_dim, HIDDEN1_SIZE),
        )
        * jnp.sqrt(1.0 / inp_dim),
        "w2": jax.random.normal(
            w2_rng,
            (HIDDEN1_SIZE, HIDDEN2_SIZE),
        )
        * jnp.sqrt(1.0 / HIDDEN1_SIZE),
        "w3": jax.random.normal(
            w3_rng,
            (HIDDEN2_SIZE, out_dim),
        )
        * jnp.sqrt(1.0 / HIDDEN2_SIZE),
        "h1": jax.random.normal(
            t1_rng,
            (HIDDEN1_SIZE,),
        )
        * scale_s,
        "h2": jax.random.normal(
            t2_rng,
            (HIDDEN2_SIZE,),
        )
        * scale_s,
        "h3": jax.random.normal(
            t3_rng,
            (out_dim,),
        )
        * scale_s,
        "b1": jnp.zeros((HIDDEN1_SIZE,)),
        "b2": jnp.zeros((HIDDEN2_SIZE,)),
        "b3": jnp.zeros((out_dim,)),
    }
