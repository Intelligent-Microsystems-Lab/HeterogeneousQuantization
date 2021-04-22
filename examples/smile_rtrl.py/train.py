# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import pickle
import time
import datetime
import sys

from absl import app
from absl import flags
from absl import logging

import jax
import jax.numpy as jnp


from flax.metrics import tensorboard
import matplotlib.pyplot as plt

sys.path.append("../..")
from model import init_state, core_fn, output_fn, init_params  # noqa: E402
from rtrl import get_rtrl_grad_func  # noqa: E402

# from jax.config import config
# config.update('jax_disable_jit', True)

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/smile_rtrl-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)
INPUT_FILE = flags.DEFINE_string(
    "input_file", "../../datasets/smile/input_700_250_25.pkl", ""
)
TARGET_FILE = flags.DEFINE_string(
    "target_file", "../../datasets/smile/smile95.pkl", ""
)
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.1, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.1, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 100_000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 10, "")
HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 64, "")
SEED = flags.DEFINE_integer("seed", 42, "")


def mse_loss(logits, labels, mask):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    return loss


rtrl_grad_fn = get_rtrl_grad_func(core_fn, output_fn, mse_loss, False)


@jax.jit
def train_step(params, batch):
    out_dim = batch["target_seq"].shape[1]

    init_s = init_state(out_dim, 1, HIDDEN_SIZE.value)

    (loss_val, (final_state, output_seq)), (
        core_grads,
        output_grads,
    ) = rtrl_grad_fn(params["cf"], params["of"], init_s, batch)

    # simple SGD step
    params["cf"] = jax.tree_multimap(
        lambda x, y: x - LEARNING_RATE.value * y / 250,
        params["cf"],
        core_grads,
    )
    params["of"] = jax.tree_multimap(
        lambda x, y: x - LEARNING_RATE.value * y / 250,
        params["of"],
        output_grads,
    )

    return params, loss_val / 250, output_seq


def get_data():
    """Load smile dataset into memory."""
    with open(INPUT_FILE.value, "rb") as f:
        x_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

    with open(TARGET_FILE.value, "rb") as f:
        y_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

    x_train = jnp.moveaxis(x_train, (0, 1, 2), (1, 0, 2))
    y_train = jnp.moveaxis(y_train, (0, 1, 2), (1, 0, 2))

    return {"input_seq": x_train, "target_seq": y_train, "mask_seq": None}


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)
    data = get_data()

    inp_dim = data["input_seq"].shape[2]
    out_dim = data["target_seq"].shape[2]

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(
        p_rng, inp_dim, out_dim, INIT_SCALE_S.value, HIDDEN_SIZE.value
    )

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())

    t_loop_start = time.time()
    for step in range(TRAINING_STEPS.value):
        params, loss_val, logits = train_step(params, data)

        summary_writer.scalar("train_loss", loss_val, (step + 1))
        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        t_loop_start = time.time()

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            logging.info("step: %d, loss: %.4f", step + 1, loss_val)

            # save picture
            plt.clf()
            fig, axes = plt.subplots(nrows=1, ncols=3)
            axes[0].matshow(data["target_seq"][:, 0, :].transpose())
            axes[1].matshow(logits[:, 0, :].transpose())
            axes[2].matshow(
                data["target_seq"][:, 0, :].transpose()
                - logits[:, 0, :].transpose()
            )
            plt.tight_layout()
            plt.savefig(WORK_DIR.value + "/smile.png")
            plt.close()


if __name__ == "__main__":
    app.run(main)
