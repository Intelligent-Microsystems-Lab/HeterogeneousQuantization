# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
#
# based on https://github.com/BerenMillidge/PredictiveCodingBackprop and
# "Practical Real Time Recurrent Learning with a Sparse Approximation to the
# Jacobian", 2020

import jax
import jax.numpy as jnp

from model import core_fn, output_fn


def relu_deriv(x):
    xrel = jax.nn.relu(x)
    return jax.lax.select(xrel > 0, jnp.ones_like(x), jnp.zeros_like(x))


def linear_deriv(x):
    return jnp.ones_like(x)


def forward_sweep(params, inpt, state, infl_m):
    # step for core fn
    _, h = core_fn(params["cf"], state, inpt)

    # Compute jacobians of state w.r.t. prev state and params.
    jac_fn = jax.jacrev(lambda p, s, i: core_fn(p, s, i)[1], argnums=(0, 1))
    batched_jac_fn = jax.vmap(lambda s, i: jac_fn(params["cf"], s, i))
    p_jac, s_jac = batched_jac_fn(state, inpt)

    # Compute new influence matrix
    new_infl = jax.tree_multimap(
        lambda j_i, infl_i: j_i
        + jnp.einsum("bHh,bh...->bH...", s_jac, infl_i),
        p_jac,
        infl_m,
    )

    # compute output
    y_pred, _ = output_fn(params["of"], None, h)

    return h, y_pred, new_infl


def infer(
    params,
    inpt,
    targt,
    y_pred,
    h_pred,
    n_inference_steps,
    inference_lr,
):
    # matching inputs and targets here especially inpt, target ...

    hs = h_pred

    for n in range(n_inference_steps):
        e_ys = targt - y_pred
        e_hs = hs - h_pred
        hdelta = e_hs - jnp.dot(
            e_ys * linear_deriv(jnp.dot(h_pred, params["of"]["wo"])),
            params["of"]["wo"].transpose(),
        )  # derivative of output
        fn_deriv = relu_deriv(
            jnp.dot(h_pred, params["cf"]["h1"])
            + jnp.dot(inpt, params["cf"]["w1"])
        )  # derivatie of hidden layer
        hdelta -= jnp.dot((e_hs * fn_deriv), params["cf"]["h1"].transpose())
        hs -= inference_lr * hdelta

    return e_ys, e_hs


# I took away the mask here because well we dont really use it....
def compute_grads(params, inpt, h_inpt, e_ys, e_hs, h_pred, infl_x, infl_h):
    dWy = jnp.dot(
        h_pred.transpose(),
        (-1 * e_ys * linear_deriv(jnp.dot(h_pred, params["of"]["wo"]))),
    )

    dWx = jnp.einsum("bH...,bH->...", infl_x, -1 * e_hs)
    dWh = jnp.einsum("bH...,bH->...", infl_h, -1 * e_hs)

    return {
        "cf": {"w1": dWx, "h1": dWh},
        "of": {
            "wo": dWy,
        },
    }


def make_zero_infl(param_exemplar, state_exemplar):
    def make_infl_for_one_state(t):
        return jax.tree_map(
            lambda p: jnp.zeros(shape=list(t.shape) + list(p.shape)),
            param_exemplar,
        )

    infl = jax.tree_map(make_infl_for_one_state, state_exemplar)
    return infl


def grad_compute(params, batch, state, n_inference_steps, inference_lr):
    def rtrl_pc_scan_fn(carry, x):
        (
            state,
            infl_acc,
            grad_acc,
            loss_acc,
        ) = carry

        inpt, targt, msk = x

        h_pred, y_pred, new_infl = forward_sweep(params, inpt, state, infl_acc)

        e_ys, e_hs = infer(
            params,
            inpt,
            targt,
            y_pred,
            h_pred,
            n_inference_steps,
            inference_lr,
        )

        grads = compute_grads(
            params,
            inpt,
            state,
            e_ys,
            e_hs,
            h_pred,
            new_infl["w1"],
            new_infl["h1"],
        )

        new_grad_acc = jax.tree_multimap(
            lambda x, y: jnp.add(x, y) * msk[0], grad_acc, grads
        )

        new_carry = (
            h_pred,
            new_infl,
            new_grad_acc,
            loss_acc + jnp.mean((targt - y_pred) ** 2),
        )

        return new_carry, y_pred

    zero_infl = make_zero_infl(params["cf"], state)
    zero_grad = jax.tree_map(jnp.zeros_like, params)

    final_carry, output_seq = jax.lax.scan(
        rtrl_pc_scan_fn,
        init=(state, zero_infl, zero_grad, 0.0),
        xs=(batch["input_seq"], batch["target_seq"], batch["mask_seq"]),
    )

    return final_carry[2], output_seq, final_carry[3]
