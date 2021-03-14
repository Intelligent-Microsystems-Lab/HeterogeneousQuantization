"""Copy task example.
Library file which executes the training and evaluation loop for copy task
example with bptt.

Vanilla RNN from https://github.com/google-research/computation-thru-dynamics/
blob/master/integrator_rnn_tutorial/rnn.py
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
    return jnp.where(x > 0, x, 0) - x * z + jnp.log(1 + jnp.exp(-jnp.abs(x)))


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


def error_bit_char(logits, target):
    total_bits = jnp.prod(jnp.array(logits.shape))
    error = jnp.sum(~(jnp.round(logits) == target)) / total_bits
    return error * 8 # bits/char


def train_step(
    optimizer,
    batch,
    config,
):
    """Train for a single step."""

    def loss_fn(params):
        hidden_state, ts_output = batched_rnn_run(
            optimizer.target, batch["seq"]
        )

        loss = masked_sigmoid_cross_entropy(
            ts_output, batch["target"], batch["mask"]
        )
        # for visualization
        ts_output = jnp.expand_dims(batch["mask"], -1) * nn.sigmoid(ts_output)
        return loss, ts_output

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val, ts_output), grad = grad_fn(optimizer.target)
    optimizer = optimizer.apply_gradient(grad)

    error = error_bit_char(ts_output, batch["target"])
    return optimizer, loss_val, ts_output, error


def random_vrnn_params(key, u, n, o, g=1.0):
    """Generate random RNN parameters"""

    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    hscale = 0.001
    ifactor = g / jnp.sqrt(u)
    hfactor = g / jnp.sqrt(n)
    pfactor = g / jnp.sqrt(n)
    return {
        "h0": jax.random.normal(k1, (n,)) * hscale,
        "wI": jax.random.normal(k2, (n, u)) * ifactor,
        "wR": jax.random.normal(k3, (n, n)) * hfactor,
        "wO": jax.random.normal(k4, (o, n)) * pfactor,
        "bR": jnp.zeros([n]),
        "bO": jnp.zeros([o]),
    }


def affine(params, x):
    """Implement y = w x + b"""
    return jnp.dot(params["wO"], x) + params["bO"]


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims) So
# map over first dimension to hand t_x_m.  I.e. if affine yields
# n_y_1 = dot(n_W_m, m_x_1), then batch_affine yields t_y_n.
batch_affine = jax.vmap(affine, in_axes=(None, 0))


def vrnn(params, h, x):
    """Run the Vanilla RNN one step"""
    a = jnp.dot(params["wI"], x) + params["bR"] + jnp.dot(params["wR"], h)
    return jnp.tanh(a)


def vrnn_scan(params, h, x):
    """Run the Vanilla RNN one step, returning (h ,h)."""
    h = vrnn(params, h, x)
    return h, h


def vrnn_run_with_h0(params, x_t, h0):
    """Run the Vanilla RNN T steps, where T is shape[0] of input."""
    h = h0
    f = functools.partial(vrnn_scan, params)
    _, h_t = jax.lax.scan(f, h, x_t, unroll=20)
    o_t = batch_affine(params, h_t)
    return h_t, o_t


def vrnn_run(params, x_t):
    """Run the Vanilla RNN T steps, where T is shape[0] of input."""
    return vrnn_run_with_h0(
        params, x_t, params["h0"]
    )  # jnp.zeros(params['bR'].shape[0],)


# Let's upgrade it to handle batches using `vmap`
# Make a batched version of the `predict` function
batched_rnn_run = jax.vmap(vrnn_run, in_axes=(None, 0))
batched_rnn_run_w_h0 = jax.vmap(vrnn_run_with_h0, in_axes=(None, 0, 0))


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

    initial_params = random_vrnn_params(
        init_rng1,
        u=config.num_bits + 2,
        n=config.hidden_units,
        o=config.num_bits + 1,
    )

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    optimizer = create_optimizer(initial_params, config.learning_rate)

    j_train_step = jax.jit(
        functools.partial(train_step, config=config)
    )  # functools.partial(train_step, config=config)#

    total_loss = 0
    #T = 1
    t_loop_start = time.time()
    for step in range(config.num_steps):
        sample = dataset._build()

        rng, step_rng = jax.random.split(rng)
        optimizer, loss, output, error = j_train_step(
            optimizer,
            batch={
                "seq": sample.observations,
                "target": sample.target,
                "mask": sample.mask,
            },
        )

        # if error < .15 :
        #     print("Increase T")
        #     T += 1
        #     rng, cur_rng = jax.random.split(rng, 2)
        #     dataset = repeat_copy.RepeatCopy(
        #         cur_rng,
        #         config.num_bits,
        #         config.batch_size,
        #         config.min_length,
        #         T,
        #         config.min_repeats,
        #         config.max_repeats,
        #     )

        total_loss += loss
        if (step + 1) % config.report_interval == 0:
            dataset_string = dataset.to_human_readable(
                sample, jnp.round(output)
            )
            logging.info(
                "%d: Avg training loss %f. last error: %f \n%s",
                step + 1,
                total_loss / config.report_interval,
                error,
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
