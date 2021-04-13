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
import time

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

sys.path.append("../..")
from datasets import copy_task
from model import *

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir", "../../../training_dir/params_bptt_copy_task/", ""
)
BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.2, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 1_000_000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 10, "")
SEED = flags.DEFINE_integer("seed", 42, "")

NUM_BITS = 6  # dataset specific value


def compute_metrics(logits, labels, mask):
    # mask irrelevant outputs
    mask = jnp.repeat(jnp.expand_dims(mask, 2), NUM_BITS + 1, 2)
    logits = logits * mask
    # simple MSE loss
    loss = ((logits - labels) ** 2).sum() / mask.sum()

    accuracy = 1 - (~(jnp.round(logits) == labels)).sum() / mask.sum()

    return {"loss": loss, "accuracy": accuracy}


def mse_loss(logits, labels, mask):
    # mask irrelevant outputs
    mask = jnp.repeat(jnp.expand_dims(mask, 2), NUM_BITS + 1, 2)
    logits = logits * mask
    # simple MSE loss
    loss = ((logits - labels) ** 2).sum() / mask.sum()

    return loss


@jax.jit
def train_step(params, batch):
    local_batch_size = batch["observations"].shape[0]

    init_s = init_state(NUM_BITS + 1, local_batch_size)

    def loss_fn(params):
        nn_model_fn = functools.partial(nn_model, params)
        final_carry, output_seq = jax.lax.scan(
            nn_model_fn,
            init=init_s,
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

    return params, metrics, grad


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch["observations"].shape[0]

    nn_model_fn = functools.partial(nn_model, params)

    init_s = init_state(NUM_BITS + 1, local_batch_size)

    final_carry, output_seq = jax.lax.scan(
        nn_model_fn,
        init=init_s,
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
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(p_rng, NUM_BITS + 2, NUM_BITS + 1, INIT_SCALE_S.value)

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    params_hist = {0: params}
    t_loop_start = time.time()
    for step in range(int(TRAINING_STEPS.value / BATCH_SIZE.value) + 1):
        # Do a batch of SGD.
        rng, inpt_rng = jax.random.split(rng, 2)
        batch_idx = jax.random.choice(
            inpt_rng,
            a=train_ds["mask"].shape[0],
            shape=(BATCH_SIZE.value,),
            replace=False,
        )
        batch = {
            "observations": train_ds["observations"][batch_idx, :, :],
            "target": train_ds["target"][batch_idx, :, :],
            "mask": train_ds["mask"][batch_idx, :],
        }
        params, train_metrics, grads = train_step(params, batch)

        params_hist[step + 1] = grads

        with open(WORK_DIR.value + "/params_hist.pickle", "wb") as handle:
            pickle.dump(params_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        t_loop_start = time.time()
        for key, val in train_metrics.items():  # type: ignore
            tag = "train_%s" % key
            summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            rng, inpt_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.choice(
                inpt_rng,
                a=test_ds["mask"].shape[0],
                shape=(test_ds["mask"].shape[0],),
                replace=False,
            )
            shuffle_test_ds = {
                "observations": test_ds["observations"][batch_idx, :, :],
                "target": test_ds["target"][batch_idx, :, :],
                "mask": test_ds["mask"][batch_idx, :],
            }

            eval_metrics, ts_output = eval_model(params, shuffle_test_ds)

            logging.info(
                "step: %d, train_loss: %.4f, train_accuracy: %.4f, eval_loss: %.4f, eval_accuracy: %.4f",
                (step + 1) * BATCH_SIZE.value,
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
            )

            for key, val in eval_metrics.items():  # type: ignore
                tag = "eval_%s" % key
                summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)

            # save picture
            plt.clf()
            fig, axes = plt.subplots(nrows=3, ncols=3)
            for i in range(3):
                mask = jnp.repeat(
                    jnp.expand_dims(shuffle_test_ds["mask"][i, :], 1),
                    NUM_BITS + 1,
                    1,
                )
                axes[i, 0].matshow(
                    shuffle_test_ds["observations"][i, :, :].transpose()
                )
                axes[i, 1].matshow(
                    shuffle_test_ds["target"][i, :, :].transpose()
                )
                axes[i, 2].matshow((ts_output[:, i, :] * mask).transpose())
            plt.tight_layout()
            plt.savefig(WORK_DIR.value + "/copy_task.png")
            plt.close()


if __name__ == "__main__":
    app.run(main)
