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

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from flax.metrics import tensorboard

import repeat_copy
import uuid

# parameters
WORK_DIR = flags.DEFINE_string("work_dir", "/tmp/" + str(uuid.uuid4()), "")

TRAIN_BATCH_SIZE = flags.DEFINE_integer("train_batch_size", 256, "")
HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 256, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 1e-3, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 2_000_000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 100, "")
SEED = flags.DEFINE_integer("seed", 42, "")
UNROLL_TIME_STEPS = flags.DEFINE_integer("unroll_time_steps", 25, "")

# copy task parameters
NUM_BITS = flags.DEFINE_integer(
    "num_bits", 4, "Dimensionality of each vector to copy"
)
MIN_LENGTH = flags.DEFINE_integer(
    "min_length",
    1,
    "Lower limit on number of vectors in the observation pattern to copy",
)
MAX_LENGTH = flags.DEFINE_integer(
    "max_length",
    40,
    "Upper limit on number of vectors in the observation pattern to copy",
)
MIN_REPEATS = flags.DEFINE_integer(
    "min_repeats", 1, "Lower limit on number of copy repeats."
)
MAX_REPEATS = flags.DEFINE_integer(
    "max_repeats", 2, "Upper limit on number of copy repeats."
)


class TrainingState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


def make_network() -> hk.RNNCore:
    """Defines the network architecture."""
    model = hk.DeepRNN(
        [
            hk.VanillaRNN(HIDDEN_SIZE.value),
            hk.Linear(NUM_BITS.value + 1),
        ]
    )
    return model


def make_optimizer() -> optax.GradientTransformation:
    """Defines the optimizer."""
    return optax.adam(LEARNING_RATE.value)


def sigmoid_cross_entropy_with_logits(x, z):
    # from https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return jnp.where(x > 0, x, 0) - x * z + jnp.log(1 + jnp.exp(-jnp.abs(x)))


def inference(batch):
    core = make_network()
    batch_size, sequence_length, spatial_dim = batch.observations.shape
    initial_state = core.initial_state(batch_size)
    logits, _ = hk.dynamic_unroll(
        core, batch.observations, initial_state, time_major=False
    )

    return logits


def masked_sigmoid_cross_entropy(
    batch, time_average=True, log_prob_in_bits=True
):
    """Adds ops to graph which compute the (scalar) NLL of the target sequence.
    The logits parametrize independent bernoulli distributions per time-step
    and per batch element, and irrelevant time/batch elements are masked out by
    the mask tensor.
    Args:
      logits: `Tensor` of activations for which sigmoid(`logits`) gives the
          bernoulli parameter.
      target: time-major `Tensor` of target.
      mask: time-major `Tensor` to be multiplied elementwise with cost T x B
          cost masking out irrelevant time-steps.
      time_average: optionally average over the time dimension (sum by default)
      log_prob_in_bits: iff True express log-probabilities in bits (default
          nats).
    Returns:
      A `Tensor` representing the log-probability of the target.
    """
    batch_size, sequence_length, spatial_dim = batch.observations.shape

    logits = inference(batch)

    # https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc
    xent = sigmoid_cross_entropy_with_logits(logits, batch.target)
    loss_time_batch = jnp.sum(xent, axis=2)
    loss_batch = jnp.sum(loss_time_batch * batch.mask, axis=0)

    if time_average:
        mask_count = jnp.sum(batch.mask, axis=0)
        loss_batch /= mask_count + jnp.finfo(jnp.float32).eps

    loss = jnp.sum(loss_batch) / batch_size
    if log_prob_in_bits:
        loss /= jnp.log(2.0)

    return loss


@jax.jit
def update(state: TrainingState, batch: Any) -> TrainingState:
    """Does a step of SGD given inputs & targets."""
    _, optimizer = make_optimizer()
    _, loss_fn = hk.without_apply_rng(
        hk.transform(masked_sigmoid_cross_entropy)
    )
    loss_val, gradients = jax.value_and_grad(loss_fn)(state.params, batch)
    updates, new_opt_state = optimizer(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return loss_val, TrainingState(params=new_params, opt_state=new_opt_state)


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    flags.FLAGS.alsologtostderr = True
    rng = hk.PRNGSequence(SEED.value)

    dataset = repeat_copy.RepeatCopy(
        next(rng),
        NUM_BITS.value,
        TRAIN_BATCH_SIZE.value,
        MIN_LENGTH.value,
        1,
        MIN_REPEATS.value,
        MAX_REPEATS.value,
    )

    # Make loss, sampler, and optimizer.
    _, inference_fn = hk.without_apply_rng(hk.transform(inference))
    params_init, loss_fn = hk.without_apply_rng(
        hk.transform(masked_sigmoid_cross_entropy)
    )
    opt_init, _ = make_optimizer()

    loss_fn = jax.jit(loss_fn)
    inference_fn = jax.jit(inference_fn)

    # Initialize training state.
    rng = hk.PRNGSequence(SEED.value)
    initial_params = params_init(next(rng), dataset._build())
    initial_opt_state = opt_init(initial_params)
    state = TrainingState(params=initial_params, opt_state=initial_opt_state)

    # Training loop.
    total_loss = 0
    T = 1
    for step in range(TRAINING_STEPS.value + 1):
        # Do a batch of SGD.
        train_batch = dataset._build()
        loss_val, state = update(state, train_batch)
        total_loss += loss_val

        summary_writer.scalar("T", T, step + 1)
        summary_writer.scalar("BPC", loss_val, step + 1)

        if loss_val < 0.15 and T < MAX_LENGTH.value:
            T += 1
            dataset = repeat_copy.RepeatCopy(
                next(rng),
                NUM_BITS.value,
                TRAIN_BATCH_SIZE.value,
                MIN_LENGTH.value,
                T,
                MIN_REPEATS.value,
                MAX_REPEATS.value,
            )

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            logits = inference_fn(state.params, train_batch)
            ts_output = jnp.expand_dims(train_batch.mask, -1) * jax.nn.sigmoid(
                logits
            )
            logging.info(
                {
                    "T": T,
                    "step": step + 1,
                    "loss": float(total_loss / EVALUATION_INTERVAL.value),
                }
            )
            print(
                dataset.to_human_readable(
                    train_batch, model_output=jnp.round(ts_output)
                )
            )
            total_loss = 0


if __name__ == "__main__":
    app.run(main)
