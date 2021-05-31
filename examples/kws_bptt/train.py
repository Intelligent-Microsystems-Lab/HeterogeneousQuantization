# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Dataset from https://www.research.ibm.com/dvsgesture/

from absl import app
from absl import flags
from absl import logging
from itertools import islice
import functools
import time
import datetime

import jax
import jax.numpy as jnp

import sys

from flax.metrics import tensorboard
from flax.training import common_utils
from flax import optim
from flax.training.lr_schedule import create_cosine_learning_rate_schedule

import torch
import tensorflow as tf
from dataloader import SpeechCommandsGoogle

# from jax.config import config
# config.update("jax_disable_jit", True)


sys.path.append("../..")
from model import (
    init_state,
    init_params,
    nn_model,
)  # noqa: E402

# parameters
SEED = flags.DEFINE_integer("seed", 42, "")
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/kws_bptt-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)

TRAINING_STEPS = flags.DEFINE_integer("training_epochs", 40000, "")
WARMUP_STEPS = flags.DEFINE_integer("warmup_epochs", 100, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 100, "")

HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 1024, "")

BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.01, "")
MOMENTUM = flags.DEFINE_float("momentum", 0.9, "")
# UPDATE_FREQ = flags.DEFINE_integer("update_freq", 100, "")
# GRAD_ACCUMULATE = flags.DEFINE_bool("grad_accumulate", True, "")
# GRAD_CLIP = flags.DEFINE_float("grad_clip", 100.0, "")

NUM_CLASSES = 12
MFCC_DIM = 40
DTYPE = jnp.float32


def cross_entropy_loss(logits, targt):
    # loss function over full time
    logits = jax.nn.sigmoid(logits)

    targt = jax.nn.one_hot(targt, num_classes=NUM_CLASSES)
    logits = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def compute_metrics(logits, labels):
    # simple MSE loss
    loss = cross_entropy_loss(logits, labels)

    accuracy = jnp.mean(jnp.argmax(logits.sum(0), axis=-1) == labels)

    return {"loss": loss, "accuracy": accuracy}


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(step, optimizer, lr_fn, batch):
    local_batch_size = batch["audio"].shape[1]
    init_s = init_state(MFCC_DIM, local_batch_size, HIDDEN_SIZE.value, DTYPE)

    def loss_fn(params):
        nn_model_fn = functools.partial(nn_model, params)
        final_carry, output_seq = jax.lax.scan(
            nn_model_fn, init=init_s, xs=batch["audio"]
        )
        loss = cross_entropy_loss(output_seq, batch["label"])
        return loss, output_seq

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logits), grad = grad_fn(optimizer.target)

    lr = lr_fn(step)
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    metrics = compute_metrics(logits, batch["label"])

    return optimizer, metrics


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch["audio"].shape[1]
    nn_model_fn = functools.partial(nn_model, params)
    init_s = init_state(MFCC_DIM, local_batch_size, HIDDEN_SIZE.value, DTYPE)

    final_carry, output_seq = jax.lax.scan(
        nn_model_fn, init=init_s, xs=batch["audio"]
    )
    metrics = compute_metrics(output_seq, batch["label"])

    return metrics


def preprocess(batch):
    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=batch["audio"].t(),
        window_size=640,
        stride=320,
        magnitude_squared=True,
    )
    batch["audio"] = jnp.moveaxis(
        jnp.array(
            tf.raw_ops.Mfcc(
                spectrogram=spectrogram,
                sample_rate=16000,
                upper_frequency_limit=4000,
                lower_frequency_limit=20,
                filterbank_channel_count=40,
                dct_coefficient_count=MFCC_DIM,
                name=None,
            )
        ),
        (0, 1, 2),
        (1, 0, 2),
    )
    batch["label"] = jnp.array(batch["label"])

    return batch


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)

    speech_dataset_train = SpeechCommandsGoogle(
        root_dir="data/speech_commands_v0.02",
        train_test_val="training",
        batch_size=BATCH_SIZE.value,
        epochs=TRAINING_STEPS.value,
    )

    speech_dataset_val = SpeechCommandsGoogle(
        root_dir="data/speech_commands_v0.02",
        train_test_val="validation",
        batch_size=BATCH_SIZE.value,
        epochs=TRAINING_STEPS.value,
    )

    speech_dataset_test = SpeechCommandsGoogle(
        root_dir="data/speech_commands_test_set_v0.02",
        train_test_val="testing",
        batch_size=BATCH_SIZE.value,
        epochs=TRAINING_STEPS.value,
    )

    train_dataloader = torch.utils.data.DataLoader(
        speech_dataset_train,
        batch_size=BATCH_SIZE.value,
        shuffle=True,
        num_workers=4,
    )
    test_dataloader = torch.utils.data.DataLoader(
        speech_dataset_test,
        batch_size=BATCH_SIZE.value,
        shuffle=True,
        num_workers=4,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        speech_dataset_val,
        batch_size=BATCH_SIZE.value,
        shuffle=True,
        num_workers=4,
    )

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(p_rng, MFCC_DIM, NUM_CLASSES, 1.0, HIDDEN_SIZE.value)

    optimizer = optim.Momentum(beta=MOMENTUM.value, nesterov=True).create(
        params
    )
    learning_rate_fn = create_cosine_learning_rate_schedule(
        LEARNING_RATE.value,
        1.0,
        TRAINING_STEPS.value,
        warmup_length=WARMUP_STEPS.value,
    )

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    t_loop_start = time.time()
    for step, batch in enumerate(
        islice(train_dataloader, TRAINING_STEPS.value)
    ):
        # Do a batch of SGD.
        batch = preprocess({"audio": batch[0], "label": batch[1]})
        optimizer, train_metrics = train_step(
            step, optimizer, learning_rate_fn, batch
        )

        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        summary_writer.scalar("lr", learning_rate_fn(step), (step + 1))
        t_loop_start = time.time()
        for key, val in train_metrics.items():  # type: ignore
            tag = "train_%s" % key
            summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)

        # Periodically report loss
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            eval_metrics = []
            for batch in validation_dataloader:
                batch = preprocess({"audio": batch[0], "label": batch[1]})
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
