from absl.testing import absltest
from absl.testing import parameterized

import functools
import numpy as np

from sparse_rtrl import get_rtrl_grad_func
from model import *


def rtrl_test_data():
    return (
        dict(
            testcase_name="5_steps",
            prob_size=5,
        ),
        dict(
            testcase_name="10_steps",
            prob_size=10,
        ),
    )


def mse_loss_rtrl(logits, labels, mask):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    return loss


def mse_loss_bptt(logits, labels, mask):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    return loss


class PredCodingTest(parameterized.TestCase):
    @parameterized.named_parameters(*rtrl_test_data())
    def test_learn_pc(
        self,
        prob_size,
    ):
        rng = jax.random.PRNGKey(42)
        rng, p_rng, i_rng, t_rng = jax.random.split(rng, 4)
        params = init_params(p_rng, prob_size, prob_size, 1)

        params = jax.tree_util.tree_map(
            lambda x: 1
            / (
                2 ** jax.random.randint(p_rng, x.shape, minval=0, maxval=7)
                * 1.0
            ),
            params,
        )

        inpt = (
            2
            ** jax.random.randint(
                i_rng, (prob_size, prob_size, prob_size), minval=0, maxval=7
            )
            * 1.0
        )
        # inpt = jnp.reshape(jnp.ones(prob_size**3), (prob_size,prob_size,prob_size)) * 1.

        targt = (
            2
            ** jax.random.randint(
                t_rng, (prob_size, prob_size, prob_size), minval=0, maxval=7
            )
            * 1.0
        )

        # targt = jnp.reshape(jnp.arange(prob_size**3), (prob_size,prob_size,prob_size)) + jnp.reshape(jnp.arange(prob_size**3), (prob_size,prob_size,prob_size)).transpose()

        init_s = init_state(prob_size, prob_size) * 1.0

        rtrl_grad_fn = get_rtrl_grad_func(
            core_fn, output_fn, mse_loss_rtrl, False
        )

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

        # rtrl
        (loss_val, (final_state, output_seq)), (
            core_grads,
            output_grads,
        ) = rtrl_grad_fn(
            params["cf"],
            params["of"],
            init_s,
            {"input_seq": inpt, "target_seq": targt, "mask_seq": None},
        )
        grad_rtrl = {"cf": core_grads, "of": output_grads}

        self.assertLessEqual(
            jnp.max(
                jnp.array(
                    jax.tree_util.tree_leaves(
                        jax.tree_util.tree_multimap(
                            lambda x, y: jnp.max(((x / prob_size) - y) / x),
                            grad_rtrl,
                            grad_bptt,
                        )
                    )
                )
            ),
            1e-6,
        )


if __name__ == "__main__":
    absltest.main()
