# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Dataset from https://www.research.ibm.com/dvsgesture/

from absl import app
from absl import flags
from absl import logging
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

# from jax.config import config
# config.update("jax_disable_jit", True)

from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import (
    create_dataloader,
)

sys.path.append("../..")
from model import (
    init_state,
    init_params,
    nn_model,
    conv_feature_extractor,
)  # noqa: E402

# parameters
SEED = flags.DEFINE_integer("seed", 42, "")
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/dvsgest_bptt-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)

TRAINING_STEPS = flags.DEFINE_integer("training_epochs", 650, "")
WARMUP_STEPS = flags.DEFINE_integer("warmup_epochs", 5, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 1, "")
EVAL_BATCH_SIZE = flags.DEFINE_integer("eval_batch_size", 16, "")

HIDDEN_SIZE = flags.DEFINE_list("hidden_size", [2048], "")

DOWNSAMPLING = flags.DEFINE_integer("downsampling", 4, "")
N_ATTENTION = flags.DEFINE_integer("n_attention", None, "")
FLATTEN_DIM = flags.DEFINE_integer("flatten_dim", 1152, "")
JITTER_STD_DEV = flags.DEFINE_float("jitter_std_dev", 0, "")
NOISE_STD_DEV = flags.DEFINE_float("noise_std_dev", 0, "")

BATCH_SIZE = flags.DEFINE_integer("batch_size", 128, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.1, "")
MOMENTUM = flags.DEFINE_float("momentum", 0.9, "")
# UPDATE_FREQ = flags.DEFINE_integer("update_freq", 100, "")
# GRAD_ACCUMULATE = flags.DEFINE_bool("grad_accumulate", True, "")
# GRAD_CLIP = flags.DEFINE_float("grad_clip", 100.0, "")

TRAIN_SEQ_LEN = flags.DEFINE_integer("train_seq_len", 500, "")
EVAL_SEQ_LEN = flags.DEFINE_integer("eval_seq_len", 1800, "")


NUM_CLASSES = 11
DIM_XY = 32
INPUT_C = 2
DTYPE = jnp.bfloat16


def cross_entropy_loss(logits, targt):
    # logits = jax.nn.sigmoid(logits)
    logits = logits.mean(0)

    logits = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def mse_loss(logits, targt):
    return jnp.mean(0.5 * (logits - targt) ** 2)


def l1_loss(logits, targt, beta=1):
    return jnp.mean(
        0.5 * ((logits - targt) ** 2) / beta * (jnp.abs(logits - targt) < beta)
        + (jnp.abs(logits - targt) - 0.5 * beta)
        * (jnp.abs(logits - targt) >= beta)
    )


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
    local_batch["input_seq"] = jnp.moveaxis(
        batch[0], (0, 1, 2, 3, 4), (1, 0, 4, 3, 2)
    )
    local_batch["target_seq"] = jnp.moveaxis(batch[1], (0, 1, 2), (1, 0, 2))

    init_s = init_state(
        FLATTEN_DIM.value, local_batch_size, HIDDEN_SIZE.value, DTYPE
    )

    def loss_fn(params):
        # pre process with conv
        inpt = conv_feature_extractor().apply(
            params["conv_fe"],
            jnp.reshape(
                local_batch["input_seq"], (-1, DIM_XY, DIM_XY, INPUT_C)
            ),
        )
        inpt = jnp.reshape(inpt, ((TRAIN_SEQ_LEN.value, local_batch_size, -1)))

        nn_model_fn = functools.partial(nn_model, params)
        final_carry, output_seq = jax.lax.scan(
            nn_model_fn, init=init_s, xs=inpt, unroll=500
        )
        loss = cross_entropy_loss(output_seq, local_batch["target_seq"])
        return loss, output_seq

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, logits), grad = grad_fn(optimizer.target)

    """
    grad = jax.tree_util.tree_map(
        lambda x: x * (jnp.max(jnp.abs(x)) < 0.25), grad
    )
    grad = jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x) if jnp.max(jnp.abs(x)) > 0.25 else x, grad
    )
    grad = jax.tree_util.tree_map(lambda x: jnp.clip(x, -0.25, 0.25), grad)
    """

    max_g = jnp.array(
        (
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(
                    jnp.max, jax.tree_util.tree_map(jnp.abs, grad)
                )
            )
        )
    ).max()

    lr = lr_fn(step)
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    metrics = compute_metrics(logits, local_batch["target_seq"])

    return optimizer, metrics, step + 1, max_g


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch[0].shape[0]

    local_batch = {}
    local_batch["input_seq"] = jnp.moveaxis(
        batch[0], (0, 1, 2, 3, 4), (1, 0, 4, 3, 2)
    )
    local_batch["target_seq"] = jnp.moveaxis(batch[1], (0, 1, 2), (1, 0, 2))

    nn_model_fn = functools.partial(nn_model, params)

    init_s = init_state(
        FLATTEN_DIM.value, local_batch_size, HIDDEN_SIZE.value, DTYPE
    )

    inpt = conv_feature_extractor().apply(
        params["conv_fe"],
        jnp.reshape(local_batch["input_seq"], (-1, DIM_XY, DIM_XY, INPUT_C)),
    )
    inpt = jnp.reshape(inpt, ((EVAL_SEQ_LEN.value, local_batch_size, -1)))

    final_carry, output_seq = jax.lax.scan(nn_model_fn, init=init_s, xs=inpt)

    metrics = compute_metrics(output_seq, local_batch["target_seq"])

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
        # jitter_train=JITTER_STD_DEV.value,
        # spatial_noise_train=NOISE_STD_DEV.value,
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
        p_rng, FLATTEN_DIM.value, NUM_CLASSES, 1.0, HIDDEN_SIZE.value
    )

    # init feaure extractor
    rng, p_rng = jax.random.split(rng, 2)
    params["conv_fe"] = conv_feature_extractor().init(
        p_rng, jnp.ones([1, DIM_XY, DIM_XY, INPUT_C]).astype(jnp.bfloat16)
    )

    # make params dtype
    params = jax.tree_map(lambda x: x.astype(DTYPE), params)

    optimizer = optim.Momentum(beta=MOMENTUM.value, nesterov=True).create(
        params
    )
    # optimizer = optim.Adam().create(params)
    steps_per_epoch = len(iter(train_ds))
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
        max_g = 0
        for batch in iter(train_ds):
            batch = [jnp.array(x).astype(jnp.bfloat16) for x in batch]
            optimizer, metrics, step_opt, gm = train_step(
                step_opt, optimizer, learning_rate_fn, batch
            )
            train_metrics.append(metrics)
            max_g = max(max_g, gm)

        train_metrics = common_utils.stack_forest(train_metrics)
        train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

        summary_writer.scalar(
            "step_time",
            (time.time() - t_loop_start),
            (step + 1) * BATCH_SIZE.value * steps_per_epoch,
        )
        summary_writer.scalar(
            "lr",
            learning_rate_fn(step * steps_per_epoch),
            (step + 1) * BATCH_SIZE.value * steps_per_epoch,
        )
        t_loop_start = time.time()
        for key, val in train_metrics.items():  # type: ignore
            tag = "train_%s" % key
            summary_writer.scalar(
                tag, val, (step + 1) * BATCH_SIZE.value * steps_per_epoch
            )

        # Periodically report loss
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            eval_metrics = []
            for batch in iter(test_ds):
                batch = [jnp.array(x).astype(jnp.bfloat16) for x in batch]
                metrics = eval_model(optimizer.target, batch)
                eval_metrics.append(metrics)

            eval_metrics = common_utils.stack_forest(eval_metrics)
            eval_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

            logging.info(max_g)
            logging.info(
                "step: %d data: %d, train_loss: %.4f, train_accuracy: %.4f,"
                " eval_loss: %.4f, eval_accuracy: %.4f",
                (step + 1),
                (step + 1) * BATCH_SIZE.value * steps_per_epoch,
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
            )

            for key, val in eval_metrics.items():  # type: ignore
                tag = "eval_%s" % key
                summary_writer.scalar(
                    tag, val, (step + 1) * BATCH_SIZE.value * steps_per_epoch
                )


if __name__ == "__main__":
    app.run(main)
