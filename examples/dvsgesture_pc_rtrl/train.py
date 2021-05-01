# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Dataset from https://github.com/deepmind/dnc

from absl import app
from absl import flags
from absl import logging
import functools
import time
import datetime

import jax
import jax.numpy as jnp

import sys

import tensorflow_datasets as tfds
from flax.metrics import tensorboard
from flax.training import common_utils

import matplotlib.pyplot as plt

# from jax.config import config
# config.update("jax_disable_jit", True)

from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *

sys.path.append("../..")
from model import init_state, init_params, nn_model  # noqa: E402
from pc_rtrl import grad_compute  # noqa: E402

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/dvsgesture_rtrl_pc-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)
BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.2, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001, "")
TRAINING_STEPS = flags.DEFINE_integer("training_epochs", 100, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 1, "")
HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 64, "")
INFERENCE_STEPS = flags.DEFINE_integer("inference_steps", 100, "")
INFERENCE_LR = flags.DEFINE_float("inference_lr", 0.1, "")
SEED = flags.DEFINE_integer("seed", 42, "")

FLATTEN_DIM = 2048


def compute_metrics(logits, labels):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    accuracy = 1 - jnp.mean(~(jnp.round(logits) == labels))

    return {"loss": loss, "accuracy": accuracy}


def mse_loss(logits, labels, mask):
    # simple MSE loss
    loss = jnp.mean((logits - labels) ** 2)

    return loss


@jax.jit
def train_step(params, batch):
    local_batch_size = batch[0].shape[0]

    local_batch = {}
    local_batch["input_seq"] = jnp.moveaxis(
        batch[0].reshape((local_batch_size, 500, -1)), (0, 1, 2), (1, 0, 2)
    )
    local_batch["target_seq"] = jnp.moveaxis(
        batch[1].reshape((local_batch_size, 500, -1)), (0, 1, 2), (1, 0, 2)
    )
    local_batch["mask_seq"] = jnp.ones(
        (
            500,
            local_batch_size,
            1,
        )
    )
    init_s = init_state(FLATTEN_DIM, local_batch_size, HIDDEN_SIZE.value)

    grads, output_seq, loss_val = grad_compute(
        params, local_batch, init_s, INFERENCE_STEPS.value, INFERENCE_LR.value
    )
    # simple SGD step
    params = jax.tree_multimap(
        lambda x, y: x - LEARNING_RATE.value * jnp.clip(y / 500, -5, 5),
        params,
        grads,
    )

    metrics = compute_metrics(
        output_seq,
        local_batch["target_seq"],
    )

    return params, metrics, output_seq


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch[0].shape[0]

    local_batch = {}
    local_batch["input_seq"] = jnp.moveaxis(
        batch[0].reshape((local_batch_size, 1800, -1)), (0, 1, 2), (1, 0, 2)
    )
    local_batch["target_seq"] = jnp.moveaxis(
        batch[1].reshape((local_batch_size, 1800, -1)), (0, 1, 2), (1, 0, 2)
    )
    local_batch["mask_seq"] = jnp.ones(
        (
            1800,
            local_batch_size,
            1,
        )
    )

    nn_model_fn = functools.partial(nn_model, params)

    init_s = init_state(FLATTEN_DIM, local_batch_size, HIDDEN_SIZE.value)

    final_carry, output_seq = jax.lax.scan(
        nn_model_fn,
        init=init_s,
        xs=local_batch["input_seq"],
    )

    metrics = compute_metrics(
        output_seq,
        local_batch["target_seq"],
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
    train_ds, test_ds = create_dataloader(
        root="data/dvs_gesture/dvs_gestures_build19.hdf5",
        batch_size=BATCH_SIZE.value,
        ds=4,
        n_events_attention=4,
        num_workers=0,
    )

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(
        p_rng,
        FLATTEN_DIM,
        11,
        INIT_SCALE_S.value,
        HIDDEN_SIZE.value,
    )

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    t_loop_start = time.time()
    for step in range(TRAINING_STEPS.value):
        # Do a batch of SGD.
        train_metrics = []
        for batch in iter(train_ds):
            batch = [jnp.array(x) for x in batch]
            params, metrics, grads = train_step(params, batch)
            train_metrics.append(metrics)

        train_metrics = common_utils.stack_forest(train_metrics)
        train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        t_loop_start = time.time()
        for key, val in train_metrics.items():  # type: ignore
            tag = "train_%s" % key
            summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            eval_metrics = []
            for batch in iter(test_ds):
                batch = [jnp.array(x) for x in batch]
                metrics, ts_output = eval_model(params, batch)
                eval_metrics.append(metrics)

            eval_metrics = common_utils.stack_forest(eval_metrics)
            eval_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

            logging.info(
                "step: %d, train_loss: %.4f, train_accuracy: %.4f, eval_loss:"
                " %.4f, eval_accuracy: %.4f",
                (step + 1),
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
            )

            for key, val in eval_metrics.items():  # type: ignore
                tag = "eval_%s" % key
                summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)


if __name__ == "__main__":
    app.run(main)
