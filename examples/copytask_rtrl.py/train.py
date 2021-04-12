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
import functools
import time

from absl import app
from absl import flags
from absl import logging

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import sys

from jax.lib import xla_bridge
from flax.metrics import tensorboard
import uuid

# from repeat_copy import RepeatCopy

# from jax.config import config
# config.update("jax_disable_jit", True)

sys.path.append("../..")
from sparse_rtrl import get_rtrl_grad_func


# parameters
WORK_DIR = flags.DEFINE_string("work_dir", "/tmp/" + str(uuid.uuid4()), "")

TRAIN_BATCH_SIZE = flags.DEFINE_integer("train_batch_size", 64, "")
HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 128, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 2_000_000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 10, "")
SEED = flags.DEFINE_integer("seed", 42, "")
NUM_BITS = flags.DEFINE_integer(
    "num_bits", 6, "Dimensionality of each vector to copy"
)
MAX_LENGTH = flags.DEFINE_integer(
    "max_length",
    30,
    "Upper limit on number of vectors in the observation pattern to copy",
)


def sigmoid_cross_entropy_with_logits(x, z):
    # from https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return jnp.where(x > 0, x, 0) - x * z + jnp.log(1 + jnp.exp(-jnp.abs(x)))


def masked_sigmoid_cross_entropy(batch, logits, mask):
    batch_size, spatial_dim = batch.shape

    # https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc
    xent = sigmoid_cross_entropy_with_logits(batch, logits)
    loss_time_batch = jnp.sum(xent, axis=1)
    loss_batch = loss_time_batch * mask

    loss = jnp.sum(loss_batch) / batch_size
    loss /= jnp.log(2.0)

    return loss


def output_fn(params, inpt):
    out = jnp.dot(inpt, params["w"]) + params["b"]
    return out


def core_fn(params, state, inpt):
    input_to_hidden = (
        jnp.dot(inpt, params["inp_hidden"]) + params["b_inp_hidden"]
    )
    hidden_to_hidden = (
        jnp.dot(state, params["hidden_hidden"]) + params["b_hidden_hidden"]
    )
    out = jax.nn.relu(input_to_hidden + hidden_to_hidden)
    return out, out


class TrainingState(NamedTuple):
    core_params: dict
    c_opt_state: optax.OptState
    output_params: dict
    o_opt_state: optax.OptState


def update(apply_fn, optim, state, batch):
    T = batch["input_seq"].shape[1]
    inital_state = jnp.zeros((TRAIN_BATCH_SIZE.value, HIDDEN_SIZE.value))

    (loss_val, (final_state, output_seq)), (
        core_grads,
        output_grads,
    ) = apply_fn(state.core_params, state.output_params, inital_state, batch)

    core_grads = jax.tree_map(lambda x: x / T, core_grads)
    output_grads = jax.tree_map(lambda x: x / T, output_grads)

    updates, c_opt_state = optim(core_grads, state.c_opt_state)
    core_params = optax.apply_updates(state.core_params, updates)

    updates, o_opt_state = optim(output_grads, state.o_opt_state)
    output_params = optax.apply_updates(state.output_params, updates)

    state = TrainingState(
        core_params=core_params,
        c_opt_state=c_opt_state,
        output_params=output_params,
        o_opt_state=o_opt_state,
    )

    return loss_val / T, output_seq, state


def main(_):
    logging.info(xla_bridge.get_backend().platform)
    logging.info(jax.host_count())

    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    flags.FLAGS.alsologtostderr = True
    rng = hk.PRNGSequence(SEED.value)

    # ds = tfds.load("copy_task")
    # ds = RepeatCopy(next(rng), batch_size=TRAIN_BATCH_SIZE.value)
    # ds_it = map(tf_to_numpy, (prep_ds(ds)))

    # Initialize training state.
    rng = hk.PRNGSequence(SEED.value)

    opt_init, optim = optax.adam(LEARNING_RATE.value)

    core_params = {
        "inp_hidden": jax.random.normal(
            next(rng),
            (NUM_BITS.value + 2, HIDDEN_SIZE.value),
        )
        * 1.0
        / jnp.sqrt(NUM_BITS.value + 2),
        "hidden_hidden": jax.random.normal(
            next(rng), (HIDDEN_SIZE.value, HIDDEN_SIZE.value)
        )
        * 1.0
        / jnp.sqrt(HIDDEN_SIZE.value),
        "b_inp_hidden": jnp.zeros(HIDDEN_SIZE.value),
        "b_hidden_hidden": jnp.zeros(HIDDEN_SIZE.value),
    }
    output_params = {
        "w": jax.random.normal(
            next(rng), (HIDDEN_SIZE.value, NUM_BITS.value + 1)
        )
        * 1.0
        / jnp.sqrt(HIDDEN_SIZE.value),
        "b": jnp.zeros(NUM_BITS.value + 1),
    }

    rtrl_grad_fn = get_rtrl_grad_func(
        core_fn, output_fn, masked_sigmoid_cross_entropy, False
    )

    c_opt_state = opt_init(core_params)
    o_opt_state = opt_init(output_params)

    import pdb

    pdb.set_trace()
    # pmap
    update_step = jax.jit(
        functools.partial(
            update,
            rtrl_grad_fn,
            optim,
        ),
    )

    state = TrainingState(
        core_params=core_params,
        c_opt_state=c_opt_state,
        output_params=output_params,
        o_opt_state=o_opt_state,
    )

    # Training loop.
    T = 1
    logging.info("Start Training")
    logging.info(WORK_DIR.value)
    t_loop_start = time.time()
    for step in range(int(TRAINING_STEPS.value / TRAIN_BATCH_SIZE.value) + 1):
        # Do a batch of SGD
        train_batch = ds._build(T)
        loss_val, logits, state = update_step(state, train_batch)

        step_sec = time.time() - t_loop_start
        t_loop_start = time.time()

        summary_writer.scalar("T", T, (step + 1) * TRAIN_BATCH_SIZE.value)
        summary_writer.scalar(
            "BPC", loss_val, (step + 1) * TRAIN_BATCH_SIZE.value
        )
        summary_writer.scalar(
            "Time", step_sec, (step + 1) * TRAIN_BATCH_SIZE.value
        )

        if loss_val < 0.15 and T < MAX_LENGTH.value:
            T += 1
            ds_it = ds._build(T)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            ts_output = jnp.expand_dims(
                train_batch["mask_seq"], -1
            ) * jax.nn.sigmoid(logits)
            logging.info(
                {
                    "T": T,
                    "step": (step + 1) * TRAIN_BATCH_SIZE.value,
                    "loss": float(loss_val),
                    "time": step_sec,
                }
            )
            # print(
            #     ds.to_human_readable(
            #         data=train_batch, model_output=jnp.round(ts_output)
            #     )
            # )
            total_loss = 0
    logging.info(WORK_DIR.value)


if __name__ == "__main__":
    app.run(main)
