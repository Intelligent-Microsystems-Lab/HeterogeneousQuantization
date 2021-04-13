from absl.testing import absltest
from absl.testing import parameterized

from sparse_rtrl import get_rtrl_grad_func
from model import *


def rtrl_test_data():
    return (
        dict(
            testcase_name="base_case",
            dummy=True,
        ),
    )


class PredCodingTest(parameterized.TestCase):
    @parameterized.named_parameters(*rtrl_test_data())
    def test_learn_pc(
        self,
        dummy,
    ):

        rng, p_rng = jax.random.split(rng, 2)
        params = init_params(p_rng, inp_dim, out_dim, INIT_SCALE_S.value)

        import pdb

        pdb.set_trace()

        init_s = init_state(out_dim, 1)

        rtrl_grad_fn = get_rtrl_grad_func(core_fn, output_fn, mse_loss, False)

        # bptt
        def loss_fn(params):
            nn_model_fn = functools.partial(nn_model, params)
            final_carry, output_seq = jax.lax.scan(
                nn_model_fn,
                init=init_s,
                xs=batch["input_seq"],
            )
            loss = mse_loss(output_seq, batch["target_seq"])
            return loss, output_seq

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_val, logits), grad = grad_fn(params)

        # rtrl

        (loss_val, (final_state, output_seq)), (
            core_grads,
            output_grads,
        ) = rtrl_grad_fn(params["cf"], params["of"], init_s, batch)

        pass


if __name__ == "__main__":
    absltest.main()
