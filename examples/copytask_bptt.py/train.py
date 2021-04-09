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

import uuid

# from jax.config import config
# config.update('jax_disable_jit', True)

sys.path.append("../..")
from datasets import copy_task

# parameters
WORK_DIR = flags.DEFINE_string("work_dir", "/tmp/" + str(uuid.uuid4()), "")
BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "")
HIDDEN1_SIZE = flags.DEFINE_integer("hidden1_size", 512, "")
HIDDEN2_SIZE = flags.DEFINE_integer("hidden2_size", 256, "")
NUM_BITS = flags.DEFINE_integer("num_bits", 6, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-4, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 100_000_000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 100, "")
SEED = flags.DEFINE_integer("seed", 42, "")


def compute_metrics(logits, labels, mask):
    # mask irrelevant outputs
    logits = logits * jnp.expand_dims(mask, 2)
    # simple MSE loss
    loss = ((logits - labels) ** 2).sum() / (
        mask.sum() * 7 + jnp.finfo(jnp.float32).eps
    )
    # mask
    accuracy = 1 - (~(jnp.clip(jnp.round(logits), 0, 1) == labels)).sum() / (
        mask.sum() * 7 + jnp.finfo(jnp.float32).eps
    )

    return {"loss": loss, "accuracy": accuracy}


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


def mse_loss(logits, labels, mask):
    # mask irrelevant outputs
    logits = logits * jnp.expand_dims(mask, 2)
    # simple MSE loss
    loss = ((logits - labels) ** 2).sum() / (
        mask.sum() * 7 + jnp.finfo(jnp.float32).eps
    )

    return loss


@jax.jit
def train_step(params, batch):
    local_batch_size = batch["observations"].shape[0]

    init_state = {
        "s1": jnp.zeros(
            (
                local_batch_size,
                HIDDEN1_SIZE.value,
            )
        ),
        "s2": jnp.zeros(
            (
                local_batch_size,
                HIDDEN2_SIZE.value,
            )
        ),
        "s3": jnp.zeros(
            (
                local_batch_size,
                NUM_BITS.value + 1,
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
        loss = mse_loss(
            output_seq,
            jnp.moveaxis(batch["target"], (0, 1, 2), (1, 0, 2)),
            jnp.moveaxis(batch["mask"], (0, 1), (1, 0)),
        )
        return loss, output_seq

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grad = grad_fn(params)

    # simple SGD step
    params = jax.tree_multimap(
        lambda x, y: x - LEARNING_RATE.value * y, params, grad
    )

    # compute metrics
    metrics = compute_metrics(
        logits,
        jnp.moveaxis(batch["target"], (0, 1, 2), (1, 0, 2)),
        jnp.moveaxis(batch["mask"], (0, 1), (1, 0)),
    )

    return params, metrics


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch["observations"].shape[0]

    nn_model_fn = functools.partial(nn_model, params)

    init_state = {
        "s1": jnp.zeros(
            (
                local_batch_size,
                HIDDEN1_SIZE.value,
            )
        ),
        "s2": jnp.zeros(
            (
                local_batch_size,
                HIDDEN2_SIZE.value,
            )
        ),
        "s3": jnp.zeros(
            (
                local_batch_size,
                NUM_BITS.value + 1,
            )
        ),
    }

    final_carry, output_seq = jax.lax.scan(
        nn_model_fn,
        init=init_state,
        xs=jnp.moveaxis(batch["observations"], (0, 1, 2), (1, 0, 2)),
    )

    metrics = compute_metrics(
        output_seq,
        jnp.moveaxis(batch["target"], (0, 1, 2), (1, 0, 2)),
        jnp.moveaxis(batch["mask"], (0, 1), (1, 0)),
    )

    return metrics, output_seq


def get_datasets():
    """Load copy_task dataset train and test datasets into memory."""
    ds_builder = tfds.builder("copy_task")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split="train", batch_size=-1)
    )
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    return train_ds, test_ds


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)
    train_ds, test_ds = get_datasets()

    # initialize parameters
    rng, w1_rng, w2_rng, s1_rng, s2_rng = jax.random.split(rng, 5)
    params = {
        "w1": jax.random.normal(
            w1_rng,
            (NUM_BITS.value + 2, HIDDEN1_SIZE.value),
        )
        / jnp.sqrt(NUM_BITS.value + 2),
        "w2": jax.random.normal(
            w2_rng,
            (HIDDEN1_SIZE.value, HIDDEN2_SIZE.value),
        )
        / jnp.sqrt(HIDDEN1_SIZE.value),
        "w3": jax.random.normal(
            w2_rng,
            (HIDDEN2_SIZE.value, NUM_BITS.value + 1),
        )
        / jnp.sqrt(HIDDEN2_SIZE.value),
        "s1": jnp.clip(
            jax.random.normal(s1_rng, (HIDDEN1_SIZE.value,)) + 0.8, -1, 1
        ),
        "s2": jnp.clip(
            jax.random.normal(s1_rng, (HIDDEN2_SIZE.value,)) + 0.8, -1, 1
        ),
        "s3": jnp.clip(
            jax.random.normal(s1_rng, (NUM_BITS.value + 1,)) + 0.8, -1, 1
        ),
    }

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    for step in range(int(TRAINING_STEPS.value / BATCH_SIZE.value) + 1):
        # Do a batch of SGD.
        rng, inpt_rng = jax.random.split(rng)
        batch_idx = jax.random.choice(
            inpt_rng,
            a=train_ds["mask"].shape[0],
            shape=(BATCH_SIZE.value,),
            replace=False,
        )

        batch = {
            "observations": train_ds["observations"][batch_idx],
            "target": train_ds["target"][batch_idx],
            "mask": train_ds["mask"][batch_idx],
        }

        params, train_metrics = train_step(params, batch)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            rng, inpt_rng = jax.random.split(rng)
            batch_idx = jax.random.choice(
                inpt_rng,
                a=test_ds["mask"].shape[0],
                shape=(test_ds["mask"].shape[0],),
                replace=False,
            )
            shuffle_test_ds = {
                "observations": test_ds["observations"][batch_idx],
                "target": test_ds["target"][batch_idx],
                "mask": test_ds["mask"][batch_idx],
            }

            eval_metrics, ts_output = eval_model(params, shuffle_test_ds)

            logging.info(
                "step: %d, loss: %.4f, accuracy: %.4f",
                step + 1,
                eval_metrics["loss"],
                eval_metrics["accuracy"],
            )

            for key, val in eval_metrics.items():  # type: ignore
                tag = "eval_%s" % key
                summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)

            print(
                copy_task.to_human_readable(
                    data=shuffle_test_ds,
                    model_output=jnp.moveaxis(
                        jnp.round(ts_output), (0, 1, 2), (1, 0, 2)
                    )
                    * jnp.expand_dims(shuffle_test_ds["mask"], 2),
                )
            )


if __name__ == "__main__":
    app.run(main)
