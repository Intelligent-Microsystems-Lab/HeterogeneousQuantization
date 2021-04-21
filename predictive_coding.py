# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
#
# based on https://github.com/BerenMillidge/PredictiveCodingBackprop

import functools

import jax
import jax.numpy as jnp

from model import core_fn, output_fn


def relu_deriv(x):
    xrel = jax.nn.relu(x)
    return jax.lax.select(xrel > 0, jnp.ones_like(x), jnp.zeros_like(x))


def linear_deriv(x):
    return jnp.ones_like(x)


def forward_sweep(input_seq, params, init_s):

    out_pred = []
    h_pred = [init_s]

    p_core_fn = functools.partial(core_fn, params["cf"])
    p_out_fn = functools.partial(output_fn, params["of"])

    _, h_pred = jax.lax.scan(
        p_core_fn,
        init=init_s,
        xs=input_seq,
    )

    _, out_pred = jax.lax.scan(
        p_out_fn,
        init=jnp.zeros((input_seq.shape[1], params["of"]["wo"].shape[1])),
        xs=h_pred,
    )

    return out_pred, jnp.vstack(
        (jnp.expand_dims(jnp.zeros_like(h_pred[0, :, :]), axis=0), h_pred)
    )


def infer(
    params,
    input_seq,
    target_seq,
    y_pred,
    h_pred,
    init_s,
    n_inference_steps,
    inference_lr,
):

    e_ys = [[] for i in range(len(target_seq))]  # ouptut prediction errors
    e_hs = [
        [] for i in range(len(input_seq))
    ]  # hidden state prediction errors

    hs = [x for x in h_pred]  # h_pred#[init_s] +

    for i, (inp, targ) in reversed(
        list(enumerate(zip(input_seq, target_seq)))
    ):
        for n in range(n_inference_steps):
            e_ys[i] = targ - y_pred[i]
            e_hs[i] = hs[i + 1] - h_pred[i + 1]
            hdelta = e_hs[i] - jnp.dot(
                e_ys[i]
                * linear_deriv(jnp.dot(h_pred[i + 1], params["of"]["wo"])),
                params["of"]["wo"].transpose(),
            )
            if i < len(target_seq) - 1:
                fn_deriv = relu_deriv(
                    jnp.dot(h_pred[i + 1], params["cf"]["h1"])
                    + jnp.dot(input_seq[i + 1], params["cf"]["w1"])
                )
                hdelta -= jnp.dot(
                    (e_hs[i + 1] * fn_deriv), params["cf"]["h1"].transpose()
                )
            hs[i + 1] -= inference_lr * hdelta

    return e_ys, e_hs


def compute_grads(params, input_seq, e_ys, e_hs, h_pred):
    dWy = jnp.zeros_like(params["of"]["wo"])
    dWx = jnp.zeros_like(params["cf"]["w1"])
    dWh = jnp.zeros_like(params["cf"]["h1"])
    for i in reversed(list(range(len(input_seq)))):
        fn_deriv = relu_deriv(
            jnp.dot(input_seq[i], params["cf"]["w1"])
            + jnp.dot(h_pred[i], params["cf"]["h1"])
        )
        dWy += jnp.dot(
            h_pred[i + 1].transpose(),
            (
                e_ys[i]
                * linear_deriv(jnp.dot(h_pred[i + 1], params["of"]["wo"]))
            ),
        )
        dWx += jnp.dot(input_seq[i].transpose(), (e_hs[i] * fn_deriv))
        dWh += jnp.dot(h_pred[i].transpose(), (e_hs[i] * fn_deriv))

    return {
        "cf": {"w1": jnp.clip(dWx, -50, 50), "h1": jnp.clip(dWh, -50, 50)},
        "of": {
            "wo": jnp.clip(dWy, -50, 50),
        },
    }
