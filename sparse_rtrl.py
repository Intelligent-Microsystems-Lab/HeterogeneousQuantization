# Copyright The Authors of "Practical Real Time Recurrent Learning with a
# Sparse Approximation to the Jacobian", 2020
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp


def get_fwd_and_update_influence_func(core_f, use_snap1_approx=False):
    """Transform core_f into a one which maintains influence jacobian
    w/ RTRL."""

    def fwd_and_update_influence(prev_infl, params, state, inpt):
        # Run the forward pass on a batch of data.
        batched_model_fn = jax.vmap(lambda s, i: core_f(params, s, i))
        f_out, state_new = batched_model_fn(state, inpt)

        # Compute jacobians of state w.r.t. prev state and params.
        jac_fn = jax.jacrev(lambda p, s, i: core_f(p, s, i)[1], argnums=(0, 1))
        batched_jac_fn = jax.vmap(lambda s, i: jac_fn(params, s, i))
        p_jac, s_jac = batched_jac_fn(state, inpt)

        # Update the influence matrix according to RTRL learning rule.
        new_infl = jax.tree_multimap(
            lambda j_i, infl_i: j_i
            + jnp.einsum("bHh,bh...->bH...", s_jac, infl_i),
            p_jac,
            prev_infl,
        )

        # SnAp-1: Keep only the entries of the influence matrix which are
        # nonzero after a single core step. This is not an efficient
        # implementation.
        if use_snap1_approx:
            onestep_infl_mask = jax.tree_map(
                lambda t: (jnp.abs(t) > 0.0).astype(jnp.float32), p_jac
            )
            new_infl = jax.tree_multimap(
                lambda matrix, mask: matrix * mask, new_infl, onestep_infl_mask
            )

        return f_out, state_new, new_infl

    return fwd_and_update_influence


def compute_gradients(influence_nest, delta):
    grads = jax.tree_map(
        lambda influence_i: jnp.einsum("bH...,bH->...", influence_i, delta),
        influence_nest,
    )
    return grads


def make_zero_infl(param_exemplar, state_exemplar):
    def make_infl_for_one_state(t):
        return jax.tree_map(
            lambda p: jnp.zeros(shape=list(t.shape) + list(p.shape)),
            param_exemplar,
        )

    infl = jax.tree_map(make_infl_for_one_state, state_exemplar)
    return infl


def get_rtrl_grad_func(core_f, output_f, loss_f, use_snap1_approx):
    """Transform functions into one which computes the gradient via RTRL."""
    fwd_and_update_influence = get_fwd_and_update_influence_func(
        core_f, use_snap1_approx=use_snap1_approx
    )

    def rtrl_grad_func(core_params, output_params, state, data):
        def rtrl_scan_func(carry, x):
            """Function which can be unrolled with jax.lax.scan."""
            # Unpack state and input.
            (
                old_state,
                infl_acc,
                core_grad_acc,
                output_grad_acc,
                loss_acc,
            ) = carry
            inpt, targt, msk = x

            # Update influence matrix.
            h_t, new_state, new_infl_acc = fwd_and_update_influence(
                infl_acc, core_params, old_state, inpt
            )

            # Compute output, loss, and backprop gradients for RNN state.
            def step_loss(ps, h, t, m):
                """Compute the loss for one RNN step."""
                y = output_f(ps, h)
                return loss_f(y, t, m), y

            step_out_and_grad_func = jax.value_and_grad(
                step_loss, has_aux=True, argnums=(0, 1)
            )
            step_out, step_grad = step_out_and_grad_func(
                output_params, h_t, targt, msk
            )
            loss_t, y_out = step_out
            output_grad_t, delta_t = step_grad

            # Update accumulated gradients.
            core_grad_t = compute_gradients(new_infl_acc, delta_t)
            new_core_grad_acc = jax.tree_multimap(
                jnp.add, core_grad_acc, core_grad_t
            )
            new_output_grad_acc = jax.tree_multimap(
                jnp.add, output_grad_acc, output_grad_t
            )

            # Repack carried state and return output.
            new_carry = (
                new_state,
                new_infl_acc,
                new_core_grad_acc,
                new_output_grad_acc,
                loss_acc + loss_t,
            )
            return new_carry, y_out

        zero_infl = make_zero_infl(core_params, state)
        zero_core_grad = jax.tree_map(jnp.zeros_like, core_params)
        zero_output_grad = jax.tree_map(jnp.zeros_like, output_params)
        final_carry, output_seq = jax.lax.scan(
            rtrl_scan_func,
            init=(state, zero_infl, zero_core_grad, zero_output_grad, 0.0),
            xs=(data["input_seq"], data["target_seq"], data["mask_seq"]),
        )
        final_state, _, core_grads, output_grads, loss = final_carry
        return (loss, (final_state, output_seq)), (core_grads, output_grads)

    return rtrl_grad_func
