"""Copy task example.
Library file which executes the training and evaluation loop for copy task
example with bptt.
"""
import functools
import time
from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import ml_collections
from flax.linen.initializers import variance_scaling

from typing import Any

import repeat_copy


Array = Any

init_method = variance_scaling(1.0, "fan_avg", "uniform")


def sigmoid_cross_entropy_with_logits(x, z):
    # from https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    return x - x * z + jnp.log(1 + jnp.exp(-jnp.abs(x)))


def masked_sigmoid_cross_entropy(
    logits, target, mask, time_average=False, log_prob_in_bits=False
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
    xent = sigmoid_cross_entropy_with_logits(logits, target)
    loss_time_batch = jnp.sum(xent, axis=2)
    loss_batch = jnp.sum(loss_time_batch * mask, axis=0)

    batch_size = logits.shape[0]

    if time_average:
        mask_count = jnp.sum(mask, axis=0)
        loss_batch /= mask_count + jnp.finfo(jnp.float32).eps

    loss = jnp.sum(loss_batch) / batch_size
    if log_prob_in_bits:
        loss /= jnp.log(2.0)

    return loss


def create_optimizer(params, learning_rate):
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


def train_step(optimizer, batch, config, rng):
    """Train for a single step."""

    def loss_fn(params):
        _, ts_output = apply_model(
            batch["seq"],
            params,
            rng,
            config.hidden_units,
            batch["target"].shape[-1],
        )
        loss = masked_sigmoid_cross_entropy(
            ts_output, batch["target"], batch["mask"]
        )
        return loss, ts_output

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, ts_output), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)

    return optimizer, loss_val, ts_output


class EncoderLSTM(nn.Module):
    @functools.partial(
        nn.transforms.scan,
        variable_broadcast="params",
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        return nn.LSTMCell()(carry, x)

    @staticmethod
    def initialize_carry(hidden_size):
        # use dummy key since default state init fn is just zeros.
        return nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (), hidden_size
        )


class LSTM_model(nn.Module):
    """Model.
    Attributes:
      hidden_size: int, the number of hidden dimensions
    """

    hidden_size: int
    out_size: int

    @nn.compact
    def __call__(self, inputs):
        """Run the model.
        Args:
          encoder_inputs: masked input sequences to encode, shaped
            `[len(input_sequence), vocab_size]`.
          decoder_inputs: masked expected decoded sequences for teacher
            forcing, shaped `[len(output_sequence), vocab_size]`.
            When sampling (i.e., `teacher_force = False`), the initial time
            step is forced into the model and samples are used for the
            following inputs. The first dimension of this tensor determines
            how many steps will be decoded, regardless of the value of
            `teacher_force`.
        Returns:
          Array of decoded logits.
        """
        encoder = EncoderLSTM()
        init_carry = encoder.initialize_carry(self.hidden_size)
        logits, predictions = encoder(init_carry, inputs)

        x = nn.Dense(features=self.out_size)(predictions)
        x = nn.sigmoid(x)
        return logits, x


def apply_model(batch, params, key, hidden_size, out_size):
    def model_fn(example):
        logits, predictions = LSTM_model(hidden_size, out_size).apply(
            {"params": params},
            example,
            rngs={"lstm": key},
        )
        return logits, predictions

    return jax.vmap(model_fn)(batch)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.
    Args:
      config: Hyper-parameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The trained optimizer.
    """

    rng = jax.random.PRNGKey(config.random_seed)
    rng, init_rng0, init_rng1, init_rng2 = jax.random.split(rng, 4)
    dataset = repeat_copy.RepeatCopy(
        init_rng0,
        config.num_bits,
        config.batch_size,
        config.min_length,
        config.max_length,
        config.min_repeats,
        config.max_repeats,
    )

    dataset_tensors = dataset._build()

    initial_params = LSTM_model(
        hidden_size=config.hidden_units,
        out_size=dataset_tensors.target.shape[-1],
    ).init(
        {"params": init_rng1, "lstm": init_rng2},
        dataset_tensors.observations[:, 0, :],
    )[
        "params"
    ]

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    optimizer = create_optimizer(initial_params, config.learning_rate)

    j_train_step = jax.jit(functools.partial(train_step, config=config))

    total_loss = 0
    t_loop_start = time.time()
    for step in range(config.num_steps):
        sample = dataset._build()

        rng, step_rng = jax.random.split(rng)
        optimizer, loss, output = j_train_step(
            optimizer,
            batch={
                "seq": sample.observations,
                "target": sample.target,
                "mask": sample.mask,
            },
            rng=step_rng,
        )
        total_loss += loss
        if (step + 1) % config.report_interval == 0:
            dataset_string = dataset.to_human_readable(
                dataset_tensors, jnp.round(output)
            )
            logging.info(
                "%d: Avg training loss %f.\n%s",
                step + 1,
                total_loss / config.report_interval,
                dataset_string,
            )
            total_loss = 0

            steps_per_sec = config.report_interval / (
                time.time() - t_loop_start
            )
            t_loop_start = time.time()
            summary_writer.scalar("steps per second", steps_per_sec, step)
            summary_writer.scalar(
                "loss", total_loss / config.report_interval, step
            )

    return optimizer
