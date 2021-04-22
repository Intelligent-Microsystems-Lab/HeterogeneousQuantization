# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Dataset from https://github.com/BerenMillidge/PredictiveCodingBackprop

from absl import app
from absl import flags
from absl import logging
import functools
import time

import jax
import jax.numpy as jnp
import sys

from flax.metrics import tensorboard

# from jax.config import config
# config.update("jax_disable_jit", True)

sys.path.append("../..")
from datasets.shakespeare.datasets import get_lstm_dataset  # noqa: E402
from model import init_state, nn_model, init_params  # noqa: E402

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir", "../../../training_dir/shakespeare_bptt/", ""
)
BATCH_SIZE = flags.DEFINE_integer("batch_size", 32, "")
HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 64, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.2, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001, "")
EPOCHS_NUM = flags.DEFINE_integer("epochs_num", 100, "")
SEQ_LEN = flags.DEFINE_integer("seq_len", 50, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 10, "")
SEED = flags.DEFINE_integer("seed", 42, "")


def compute_metrics(logits, labels):
    # simple MSE loss
    loss = ((logits - labels) ** 2).sum()
    accuracy = jnp.mean(
        jnp.argmax(labels, axis=2) == jnp.argmax(logits, axis=2)
    )

    return {"loss": loss, "accuracy": accuracy}


def mse_loss(logits, labels):
    # simple MSE loss
    loss = ((logits - labels) ** 2).sum()
    return loss


@jax.partial(jax.jit, static_argnums=[2])
def train_step(params, batch, VOCAB_SIZE):

    inpt_seq = jnp.moveaxis(
        jax.nn.one_hot(batch[0], VOCAB_SIZE), (0, 1, 2), (1, 0, 2)
    )
    targt_seq = jnp.moveaxis(
        jax.nn.one_hot(batch[1], VOCAB_SIZE), (0, 1, 2), (1, 0, 2)
    )

    local_batch_size = batch[0].shape[0]
    init_s = init_state(VOCAB_SIZE, local_batch_size, HIDDEN_SIZE.value)

    def loss_fn(params):
        nn_model_fn = functools.partial(nn_model, params)
        final_carry, output_seq = jax.lax.scan(
            nn_model_fn,
            init=init_s,
            xs=inpt_seq,
        )
        loss = mse_loss(
            output_seq,
            targt_seq,
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
        targt_seq,
    )

    return params, metrics, grad


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)

    dataset, vocab_size, char2idx, idx2char = get_lstm_dataset(
        SEQ_LEN.value, BATCH_SIZE.value
    )
    dataset = [[inp.numpy(), target.numpy()] for (inp, target) in dataset]

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(
        p_rng, vocab_size, vocab_size, INIT_SCALE_S.value, HIDDEN_SIZE.value
    )

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    t_loop_start = time.time()
    for step in range(EPOCHS_NUM.value):
        for i, batch in enumerate(dataset):
            # Do a batch of SGD.
            params, train_metrics, grads = train_step(
                params, batch, vocab_size
            )

            # Periodically report
            if (i + 1) % EVALUATION_INTERVAL.value == 0:
                logging.info(
                    "step: %d, train_loss: %.4f, train_accuracy: %.4f",
                    (step * BATCH_SIZE.value) + i + 1,
                    train_metrics["loss"],
                    train_metrics["accuracy"],
                )

        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start) / (i + 1), (step + 1)
        )
        t_loop_start = time.time()
        for key, val in train_metrics.items():  # type: ignore
            tag = "train_%s" % key
            summary_writer.scalar(tag, val, (step + 1) * BATCH_SIZE.value)


if __name__ == "__main__":
    app.run(main)
