import jax.numpy as jnp
from jax import (
    vmap,
    grad,
)
from flax.core import freeze, unfreeze


def infer_pc(x, params, act_fn, beta, it_max, var_layer):
    n_layers = len(params) + 1

    # infer function
    error = [[] for i in range(n_layers)]
    f_n = [[] for i in range(n_layers)]
    f_p = [[] for i in range(n_layers)]
    # calculate initial errors
    for i, (name, layer_params) in enumerate(params.items(), 1):
        f_n[i - 1] = act_fn(x[i - 1])
        f_p[i - 1] = vmap(grad(act_fn))(x[i - 1])
        error[i] = (
            x[i]
            - jnp.dot(f_n[i - 1], layer_params["kernel"])
            - layer_params["bias"]
        ) / var_layer[i]

    for it in range(it_max):
        # update variable nodes
        for i, (name, layer_params) in enumerate(params.items(), 1):
            if i != 1 and i != n_layers:
                g = jnp.dot(layer_params["kernel"], error[i]) * f_p[i - 1]
                x[i - 1] = x[i - 1] + beta * (-error[i - 1] + g)
        # calculate errors
        for i, (name, layer_params) in enumerate(params.items(), 1):
            f_n[i - 1] = act_fn(x[i - 1])
            f_p[i - 1] = vmap(grad(act_fn))(x[i - 1])
            error[i] = (
                x[i]
                - jnp.dot(f_n[i - 1], layer_params["kernel"])
                - layer_params["bias"]
            ) / var_layer[i]
    return x, error


def learn_pc(sin, sout, params, act_fn, beta, it_max, var_layer):
    n_layers = len(params) + 1
    v_out = var_layer[-1]
    x = [[] for i in range(n_layers)]

    x[0] = sin
    # make predictions
    for i, (name, layer_params) in enumerate(params.items(), 1):
        x[i] = (
            jnp.dot(act_fn(x[i - 1]), layer_params["kernel"])
            + layer_params["bias"]
        )
    outputs = x[-1]

    # infer
    x[-1] = sout
    x, error = infer_pc(x, params, act_fn, beta, it_max, var_layer)

    grad = unfreeze(params)

    # calculate gradients
    for i, (name, _) in enumerate(params.items()):
        grad[name]["bias"] = -v_out * error[i + 1]
        grad[name]["kernel"] = (
            -v_out * jnp.outer(error[i + 1], act_fn(x[i])).transpose()
        )

    return freeze(grad), outputs
