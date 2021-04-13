# copied from
# https://github.com/deepmind/dm-haiku/blob/master/examples/rnn/train.py
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Character-level language modelling with a recurrent network in JAX."""

from typing import Any, NamedTuple
import pickle
import time

from absl import app
from absl import flags
from absl import logging
import functools

import jax
import jax.numpy as jnp
import optax
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from flax.metrics import tensorboard

import matplotlib.pyplot as plt

import uuid

sys.path.append("../..")
from model import *
from sparse_rtrl import get_rtrl_grad_func

# from jax.config import config
# config.update('jax_disable_jit', True)

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/params_rtrl_smile" + str(uuid.uuid4()) + "/",
    "",
)
INPUT_FILE = flags.DEFINE_string(
    "input_file", "../../datasets/smile/input_700_250_25.pkl", ""
)
TARGET_FILE = flags.DEFINE_string(
    "target_file", "../../datasets/smile/smile95.pkl", ""
)
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.5, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.1, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 1000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 10, "")
SEED = flags.DEFINE_integer("seed", 42, "")


def mse_loss(logits, labels, mask):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    return loss


rtrl_grad_fn = get_rtrl_grad_func(core_fn, output_fn, mse_loss, False)


@jax.jit
def train_step(params, batch):
    out_dim = batch["target_seq"].shape[1]

    init_s = init_state(out_dim, 1)

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

    with open(
        "../../../training_dir/params_bptt/params_hist.pickle", "rb"
    ) as f:
        bptt_params = pickle.load(f)

    # get data set
    rng = jax.random.PRNGKey(SEED.value)
    data = get_data()

    inp_dim = data["input_seq"].shape[2]
    out_dim = data["target_seq"].shape[2]
    t_dim = data["input_seq"].shape[1]

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(p_rng, inp_dim, out_dim, INIT_SCALE_S.value)

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())

    params_hist = {0: params}
    t_loop_start = time.time()
    for step in range(TRAINING_STEPS.value):
        max_dev = jnp.max(
            jnp.array(
                jax.tree_util.tree_leaves(
                    jax.tree_util.tree_multimap(
                        lambda x, y: jnp.max(jnp.abs(x - y)),
                        params_hist[step],
                        bptt_params[step],
                    )
                )
            )
        )

        params, loss_val, logits = train_step(params, data)
        summary_writer.scalar("train_loss", loss_val, (step + 1))
        summary_writer.scalar("param_dev", max_dev, (step + 1))
        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        t_loop_start = time.time()
        params_hist[step + 1] = params

        with open(
            WORK_DIR.value + "/params_bptt_smile.pickle", "wb"
        ) as handle:
            pickle.dump(params_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            logging.info(
                "step: %d, loss: %.4f, diff: %e", step + 1, loss_val, max_dev
            )

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
