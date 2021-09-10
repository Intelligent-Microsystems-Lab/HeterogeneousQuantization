# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

from absl.testing import absltest
from absl.testing import parameterized

import functools
import jax
import jax.numpy as jnp
import numpy as np

from rtrl import get_rtrl_grad_func
from model import init_params, init_state, core_fn, output_fn, nn_model


def rtrl_test_data():
  return (dict(testcase_name="5_steps", prob_size=5),)


def mse_loss_rtrl(logits, labels, mask):
  # simple MSE loss
  loss = jnp.mean((logits - labels) ** 2)

  return loss


def mse_loss_bptt(logits, labels, mask):
  # simple MSE loss
  loss = jnp.mean((logits - labels) ** 2)

  return loss


class UnitTests(parameterized.TestCase):
  @parameterized.named_parameters(*rtrl_test_data())
  def test_learn_rtrl(self, prob_size):
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

    init_s = init_state(prob_size, prob_size, 64, jnp.float32) * 1.0

    rtrl_grad_fn = get_rtrl_grad_func(
        core_fn, output_fn, mse_loss_rtrl, False
    )

    # bptt
    def loss_fn(params):
      nn_model_fn = functools.partial(nn_model, params)
      final_carry, output_seq = jax.lax.scan(
          nn_model_fn, init=init_s, xs=inpt
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

    self.assertTrue(
        jnp.array(
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_multimap(
                    lambda x, y: np.allclose(x / prob_size, y),
                    grad_rtrl,
                    grad_bptt,
                )
            )
        ).all()
    )


if __name__ == "__main__":
  absltest.main()
