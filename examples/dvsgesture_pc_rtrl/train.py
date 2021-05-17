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
from flax import optim
from flax.training.lr_schedule import (
    create_cosine_learning_rate_schedule,
    create_constant_learning_rate_schedule,
)

# from jax.config import config
# config.update("jax_disable_jit", True)

from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import (
    create_dataloader,
)

sys.path.append("../..")
from model import init_state, init_params, nn_model  # noqa: E402
from pc_rtrl import (
    grad_compute,
    init_conv,
    conv_feature_extractor,
)  # noqa: E402

# parameters
SEED = flags.DEFINE_integer("seed", 42, "")
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/dvsgest_rtrl_pc-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)

TRAINING_STEPS = flags.DEFINE_integer("training_epochs", 12, "")
WARMUP_STEPS = flags.DEFINE_integer("warmup_epochs", 2, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 1, "")
EVAL_BATCH_SIZE = flags.DEFINE_integer("eval_batch_size", 8, "")

HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 64, "")

CONV_FEATURE_EXTRACTOR = flags.DEFINE_bool("conv_feature_extractor", False, "")
DOWNSAMPLING = flags.DEFINE_integer("downsampling", 4, "")
N_ATTENTION = flags.DEFINE_integer("n_attention", None, "")
FLATTEN_DIM = flags.DEFINE_integer("flatten_dim", 2048, "")
JITTER_STD_DEV = flags.DEFINE_float("jitter_std_dev", 0, "")
NOISE_STD_DEV = flags.DEFINE_float("noise_std_dev", 0, "")


INFERENCE_STEPS = flags.DEFINE_integer("inference_steps", 100, "")
INFERENCE_LR = flags.DEFINE_float("inference_lr", 0.01, "")

BATCH_SIZE = flags.DEFINE_integer("batch_size", 128, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.2, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001, "")
MOMENTUM = flags.DEFINE_float("momentum", 0.9, "")
UPDATE_FREQ = flags.DEFINE_integer("update_freq", 500, "")
GRAD_ACCUMULATE = flags.DEFINE_bool("grad_accumulate", True, "")
GRAD_CLIP = flags.DEFINE_float("grad_clip", 50.0, "")

TRAIN_SEQ_LEN = flags.DEFINE_integer("train_seq_len", 500, "")
EVAL_SEQ_LEN = flags.DEFINE_integer("eval_seq_len", 1800, "")


def cross_entropy_loss(logits, targt):
    logits = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def compute_metrics(logits, labels):
    # simple MSE loss
    loss = cross_entropy_loss(logits, labels)

    accuracy = 1 - jnp.mean(
        ~(jnp.round(jnp.argmax(logits, axis=2)) == jnp.argmax(labels, axis=2))
    )

    return {"loss": loss, "accuracy": accuracy}


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(step, optimizer, lr_fn, batch):
    local_batch_size = batch[0].shape[0]

    local_batch = {}
    local_batch["input_seq"] = jnp.moveaxis(batch[0], (0, 1, 2), (1, 0, 2))
    local_batch["target_seq"] = jnp.moveaxis(batch[1], (0, 1, 2), (1, 0, 2))
    local_batch["mask_seq"] = jnp.ones(
        (
            TRAIN_SEQ_LEN.value,
            local_batch_size,
            1,
        )
    )

    init_s = init_state(FLATTEN_DIM.value, local_batch_size, HIDDEN_SIZE.value)

    optimizer, output_seq, step = grad_compute(
        step,
        optimizer,
        lr_fn,
        local_batch,
        init_s,
        INFERENCE_STEPS.value,
        INFERENCE_LR.value,
        UPDATE_FREQ.value,
        GRAD_ACCUMULATE.value,
        GRAD_CLIP.value,
        static_conv_feature_extractor=CONV_FEATURE_EXTRACTOR.value,
    )

    metrics = compute_metrics(
        output_seq,
        local_batch["target_seq"],
    )

    return optimizer, metrics, step


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch[0].shape[0]

    local_batch = {}
    local_batch["input_seq"] = jnp.moveaxis(batch[0], (0, 1, 2), (1, 0, 2))
    local_batch["target_seq"] = jnp.moveaxis(batch[1], (0, 1, 2), (1, 0, 2))
    local_batch["mask_seq"] = jnp.ones(
        (
            EVAL_SEQ_LEN.value,
            local_batch_size,
            1,
        )
    )

    nn_model_fn = functools.partial(nn_model, params)

    init_s = init_state(FLATTEN_DIM.value, local_batch_size, HIDDEN_SIZE.value)

    # pre process with conv
    if CONV_FEATURE_EXTRACTOR.value:
        inpt, _ = conv_feature_extractor(
            params, jnp.reshape(local_batch["input_seq"], (-1, 2, 128, 128))
        )

    local_batch["input_seq"] = jnp.reshape(
        local_batch["input_seq"], (EVAL_SEQ_LEN.value, local_batch_size, -1)
    )

    final_carry, output_seq = jax.lax.scan(
        nn_model_fn,
        init=init_s,
        xs=local_batch["input_seq"],
    )

    metrics = compute_metrics(
        output_seq,
        local_batch["target_seq"],
    )

    return metrics


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)
    train_ds, _ = create_dataloader(
        root="data/dvs_gesture/dvs_gestures_build19.hdf5",
        batch_size=BATCH_SIZE.value,
        ds=DOWNSAMPLING.value,
        n_events_attention=N_ATTENTION.value,
        num_workers=0,
        jitter_train=JITTER_STD_DEV.value,
        spatial_noise_train=NOISE_STD_DEV.value,
    )
    _, test_ds = create_dataloader(
        root="data/dvs_gesture/dvs_gestures_build19.hdf5",
        batch_size=EVAL_BATCH_SIZE.value,
        ds=DOWNSAMPLING.value,
        n_events_attention=N_ATTENTION.value,
        num_workers=0,
    )

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(
        p_rng,
        FLATTEN_DIM.value,
        11,
        INIT_SCALE_S.value,
        HIDDEN_SIZE.value,
    )

    # init feaure extractor
    if CONV_FEATURE_EXTRACTOR.value:
        rng, p_rng = jax.random.split(rng, 2)
        params = init_conv(p_rng, params)

    optimizer = optim.Momentum(beta=MOMENTUM.value, nesterov=True).create(
        params
    )
    steps_per_epoch = len(iter(train_ds)) * (
        TRAIN_SEQ_LEN.value / UPDATE_FREQ.value
    )
    learning_rate_fn = create_cosine_learning_rate_schedule(
        LEARNING_RATE.value,
        steps_per_epoch,
        TRAINING_STEPS.value,
        warmup_length=WARMUP_STEPS.value,
    )

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    t_loop_start = time.time()
    for step in range(TRAINING_STEPS.value):
        # Do a batch of SGD.
        train_metrics = []
        step_opt = 0
        for batch in iter(train_ds):
            batch = [jnp.array(x.bool()) for x in batch]
            optimizer, metrics, step_opt = train_step(
                step_opt, optimizer, learning_rate_fn, batch
            )
            train_metrics.append(metrics)

        train_metrics = common_utils.stack_forest(train_metrics)
        train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        summary_writer.scalar(
            "lr", learning_rate_fn(step * steps_per_epoch), (step + 1)
        )
        t_loop_start = time.time()
        for key, val in train_metrics.items():  # type: ignore
            tag = "train_%s" % key
            summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)

        # Periodically report loss
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            eval_metrics = []
            for batch in iter(test_ds):
                batch = [jnp.array(x.bool()) for x in batch]
                metrics = eval_model(optimizer.target, batch)
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
