# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer


from absl.testing import absltest
import functools

import jax
import jax.numpy as jnp
import numpy as np

from model import init_params, init_state, nn_model
from pc_rtrl import (
    grad_compute,
    conv_feature_extractor,
    conv_feature_extractor_bwd,
    init_conv,
)


# parameters
LEARNING_RATE = 0.0005
VOCAB_SIZE = 65
INIT_SCALE_S = 0.1
BATCH_SIZE = 64
HIDDEN_SIZE = 256
INFERENCE_STEPS = 10
INFERENCE_LR = 0.1


def mse_loss_bptt(logits, labels, mask):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    return loss


def simple_loss(logits, labels, mask):
    # simple loss
    loss = jnp.sum(jnp.abs(logits - labels))

    return loss


class UnitTests(absltest.TestCase):
    def test_conv_feature_extractor(self):

        rng = jax.random.PRNGKey(42)
        rng, p_rng, i_rng, t_rng = jax.random.split(rng, 4)

        params = {}
        params = init_conv(p_rng, params)

        inpt = (
            2
            ** jax.random.randint(i_rng, (21, 2, 128, 128), minval=0, maxval=7)
            * 1.0
        )

        targt = (
            2 ** jax.random.randint(t_rng, (21, 32, 4, 4), minval=0, maxval=7)
            * 1.0
        )

        # bp grads
        def loss_fn(params):
            logits, act_tracker = conv_feature_extractor(params, inpt)
            loss = simple_loss(logits, targt, None)
            return loss, None

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_val, _), grad_bp = grad_fn(params)

        # custom bp implementation
        logits, act_tracker = conv_feature_extractor(params, inpt)
        act_tracker.insert(0, inpt)
        errors = logits - targt
        g = errors / jnp.abs(errors)
        grads = conv_feature_extractor_bwd({}, params, act_tracker, g)

        # assert equality here with tight tolerances
        jax.tree_multimap(
            lambda x, y: np.testing.assert_allclose(x, y), grads, grad_bp
        )

    def test_pc_rtrl_vs_bptt(self):
        prob_size = 15
        rng = jax.random.PRNGKey(42)
        rng, p_rng, i_rng, t_rng = jax.random.split(rng, 4)

        params = init_params(p_rng, prob_size, prob_size, 1, 64)
        params = jax.tree_map(
            lambda x: (
                jnp.reshape(jnp.arange(jnp.prod(jnp.array(x.shape))), x.shape)
                / jnp.prod(jnp.array(x.shape))
                - 0.5
            )
            / 10
            if x.sum() != 0
            else x,
            params,
        )

        inpt = (
            2
            ** jax.random.randint(
                i_rng, (prob_size, prob_size, prob_size), minval=0, maxval=7
            )
            * 1.0
        )

        targt = (
            2
            ** jax.random.randint(
                t_rng, (prob_size, prob_size, prob_size), minval=0, maxval=7
            )
            * 1.0
        )

        init_s = init_state(prob_size, prob_size, 64) * 1.0

        # bptt
        def loss_fn(params):
            nn_model_fn = functools.partial(nn_model, params)
            final_carry, output_seq = jax.lax.scan(
                nn_model_fn,
                init=init_s,
                xs=inpt,
            )
            loss = mse_loss_bptt(output_seq, targt, None)
            return loss, output_seq

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_val, logits), grad_bptt = grad_fn(params)

        # pc gradients
        local_batch = {}
        local_batch["input_seq"] = jnp.moveaxis(inpt, (0, 1, 2), (1, 0, 2))
        local_batch["target_seq"] = jnp.moveaxis(targt, (0, 1, 2), (1, 0, 2))
        local_batch["mask_seq"] = jnp.ones(
            (
                15,
                BATCH_SIZE,
                1,
            )
        )
        grad_pc_rtrl, output_seq, loss_val = grad_compute(
            params,
            local_batch,
            init_s,
            INFERENCE_STEPS,
            INFERENCE_LR,
            static_conv_feature_extractor=False,
        )

        # note tolerance level here are very high
        np.testing.assert_allclose(
            grad_pc_rtrl["cf"]["w1"] / 1200,
            grad_bptt["cf"]["w1"],
            rtol=1e-0,
            atol=1e-0,
        )
        np.testing.assert_allclose(
            grad_pc_rtrl["cf"]["h1"] / 1200,
            grad_bptt["cf"]["h1"],
            rtol=1e-0,
            atol=1e-0,
        )
        np.testing.assert_allclose(
            grad_pc_rtrl["of"]["wo"] / 1200,
            grad_bptt["of"]["wo"],
            rtol=1e-0,
            atol=1e-0,
        )


if __name__ == "__main__":
    absltest.main()
