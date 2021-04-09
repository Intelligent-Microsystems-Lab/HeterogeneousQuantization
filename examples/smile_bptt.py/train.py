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

from absl import app
from absl import flags
from absl import logging
import functools

# import haiku as hk
import jax
import jax.numpy as jnp
import optax
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from flax.metrics import tensorboard

import matplotlib.pyplot as plt

import uuid

# from jax.config import config
# config.update('jax_disable_jit', True)

# parameters
WORK_DIR = flags.DEFINE_string("work_dir", "/tmp/" + str(uuid.uuid4()), "")
INPUT_FILE = flags.DEFINE_string(
    "input_file", "../../datasets/smile/input_700_250_25.pkl", ""
)
TARGET_FILE = flags.DEFINE_string(
    "target_file", "../../datasets/smile/smile95.pkl", ""
)
HIDDEN1_SIZE = flags.DEFINE_integer("hidden1_size", 512, "")
HIDDEN2_SIZE = flags.DEFINE_integer("hidden2_size", 256, "")
NUM_BITS = flags.DEFINE_integer("num_bits", 6, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 10_000_000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 200, "")
SEED = flags.DEFINE_integer("seed", 42, "")


def nn_model(params, state, x):
    # two layer feedforward statefull NN
    x = jnp.dot(x, params["w1"]) + state["s1"] * params["s1"]
    state["s1"] = x
    x = jax.nn.sigmoid(x)

    x = jnp.dot(x, params["w2"]) + state["s2"] * params["s2"]
    state["s2"] = x
    x = jax.nn.sigmoid(x)

    x = jnp.dot(x, params["w3"]) + state["s3"] * params["s3"]
    state["s3"] = x
    x = jax.nn.sigmoid(x)

    return state, x


def mse_loss(logits, labels):
    # simple MSE loss
    loss = jnp.mean((logits[:, 0, :] - labels[0, :, :]) ** 2)

    return loss


@jax.jit
def train_step(params, batch):
    out_dim = batch["target"].shape[1]

    init_state = {
        "s1": jnp.zeros(
            (
                1,
                HIDDEN1_SIZE.value,
            )
        ),
        "s2": jnp.zeros(
            (
                1,
                HIDDEN2_SIZE.value,
            )
        ),
        "s3": jnp.zeros(
            (
                1,
                out_dim,
            )
        ),
    }

    def loss_fn(params):
        nn_model_fn = functools.partial(nn_model, params)
        final_carry, output_seq = jax.lax.scan(
            nn_model_fn,
            init=init_state,
            xs=jnp.moveaxis(batch["observations"], (0, 1, 2), (1, 0, 2)),
        )
        loss = mse_loss(output_seq, batch["target"])
        return loss, output_seq

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logits), grad = grad_fn(params)

    # simple SGD step
    params = jax.tree_multimap(
        lambda x, y: x - LEARNING_RATE.value * y, params, grad
    )

    return params, loss_val, logits


def get_data():
    """Load smile dataset into memory."""
    with open(INPUT_FILE.value, "rb") as f:
        x_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

    with open(TARGET_FILE.value, "rb") as f:
        y_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

    return {"observations": x_train, "target": y_train}


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)
    data = get_data()

    inp_dim = data["observations"].shape[2]
    out_dim = data["target"].shape[2]
    t_dim = data["observations"].shape[1]

    # initialize parameters
    rng, w1_rng, w2_rng, s1_rng, s2_rng = jax.random.split(rng, 5)
    params = {
        "w1": jax.random.normal(
            w1_rng,
            (inp_dim, HIDDEN1_SIZE.value),
        )
        / jnp.sqrt(inp_dim),
        "w2": jax.random.normal(
            w2_rng,
            (HIDDEN1_SIZE.value, HIDDEN2_SIZE.value),
        )
        / jnp.sqrt(HIDDEN1_SIZE.value),
        "w3": jax.random.normal(
            w2_rng,
            (HIDDEN2_SIZE.value, out_dim),
        )
        / jnp.sqrt(HIDDEN2_SIZE.value),
        "s1": jnp.clip(
            jax.random.normal(s1_rng, (HIDDEN1_SIZE.value,)) * 0.8, -1, 1
        ),
        "s2": jnp.clip(
            jax.random.normal(s1_rng, (HIDDEN2_SIZE.value,)) * 0.8, -1, 1
        ),
        "s3": jnp.clip(jax.random.normal(s1_rng, (out_dim,)) * 0.8, -1, 1),
    }

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    for step in range(TRAINING_STEPS.value):
        params, loss_val, logits = train_step(params, data)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            logging.info("step: %d, loss: %.4f", step + 1, loss_val)

            # save picture
            plt.clf()
            fig, axes = plt.subplots(nrows=1, ncols=3)
            axes[0].matshow(data["target"][0, :, :].transpose())
            axes[1].matshow(logits[:, 0, :].transpose())
            axes[2].matshow(
                data["target"][0, :, :].transpose()
                - logits[:, 0, :].transpose()
            )
            plt.tight_layout()
            plt.savefig(WORK_DIR.value + "/smile.png")
            plt.close()


if __name__ == "__main__":
    app.run(main)
