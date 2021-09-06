# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer


from absl.testing import absltest
import functools

import jax
import jax.numpy as jnp
import numpy as np

from unit_test.datasets import get_lstm_dataset
from model import init_params, init_state, nn_model
from predictive_coding import forward_sweep, infer, compute_grads


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


@jax.jit
def train_step(params, batch, init_s):

    inpt_seq = jnp.moveaxis(
        jax.nn.one_hot(batch[0], VOCAB_SIZE), (0, 1, 2), (1, 0, 2)
    )
    targt_seq = jnp.moveaxis(
        jax.nn.one_hot(batch[1], VOCAB_SIZE), (0, 1, 2), (1, 0, 2)
    )

    out_pred, h_pred = forward_sweep(inpt_seq, params, init_s)
    e_ys, e_hs = infer(
        params,
        inpt_seq,
        targt_seq,
        out_pred,
        h_pred,
        init_s,
        INFERENCE_STEPS,
        INFERENCE_LR,
    )
    grad = compute_grads(params, inpt_seq, e_ys, e_hs, h_pred)

    return grad


class UnitTests(absltest.TestCase):
    def test_pc_equality(self):
        dataset, vocab_size, char2idx, idx2char = get_lstm_dataset(
            50, BATCH_SIZE
        )
        dataset = [[inp.numpy(), target.numpy()] for (inp, target) in dataset]

        assert vocab_size == VOCAB_SIZE

        rng = jax.random.PRNGKey(42)

        inp_dim = vocab_size
        out_dim = vocab_size

        # initialize parameters
        rng, p_rng = jax.random.split(rng, 2)
        params = init_params(
            p_rng, inp_dim, out_dim, INIT_SCALE_S, HIDDEN_SIZE
        )
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
        init_s = init_state(out_dim, 64, HIDDEN_SIZE, jnp.float32)

        grad = train_step(params, dataset[0], init_s)

        dWh = np.load("unit_test/pc_rnn_dWh.npy")
        dWx = np.load("unit_test/pc_rnn_dWx.npy")
        dWy = np.load("unit_test/pc_rnn_dWy.npy")

        self.assertTrue(
            np.allclose(
                dWx.transpose(),
                jnp.clip(grad["cf"]["w1"], -50, 50),
                rtol=1e-04,
                atol=1e-07,
            )
        )
        self.assertTrue(
            np.allclose(
                dWh.transpose(),
                jnp.clip(grad["cf"]["h1"], -50, 50),
                rtol=1e-04,
                atol=1e-07,
            )
        )
        self.assertTrue(
            np.allclose(
                dWy.transpose(),
                jnp.clip(grad["of"]["wo"], -50, 50),
                rtol=1e-03,
                atol=1e-06,
            )
        )

    def test_pc_vs_bptt(self):
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

        init_s = init_state(prob_size, prob_size, 64, jnp.float32) * 1.0

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

        # pc gradients
        out_pred, h_pred = forward_sweep(inpt, params, init_s)
        e_ys, e_hs = infer(
            params, inpt, targt, out_pred, h_pred, init_s, 100, INFERENCE_LR
        )
        grad_pc = compute_grads(params, inpt, e_ys, e_hs, h_pred)

        self.assertTrue(
            jnp.array(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_multimap(
                        lambda x, y: np.allclose(
                            x / (prob_size * 100),
                            -1.0 * y,
                            rtol=1e-0,
                            atol=1e-0,
                        ),
                        grad_pc,
                        grad_bptt,
                    )
                )
            ).all()
        )


if __name__ == "__main__":
    absltest.main()
