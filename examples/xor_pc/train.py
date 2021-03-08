"""XOR example.
Library file which executes the training and evaluation loop for XOR.
"""
import functools
from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from typing import Any
import jax
import jax.numpy as jnp
import ml_collections
from flax.linen.initializers import variance_scaling

import sys

sys.path.append("../..")
from predictive_coding import learn_pc  # noqa: E402

Array = Any

init_method = variance_scaling(1.0, "fan_avg", "uniform")


class Net(nn.Module):
    """A simple model."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Description of a forward pass
        Args:
            x: an array (inputs)
            act_fn: activation function applied after MVM
        Returns:
            An array containing the result.
        """
        x = nn.tanh(x)
        x = nn.Dense(features=5, kernel_init=init_method)(x)

        x = nn.tanh(x)
        x = nn.Dense(features=1, kernel_init=init_method)(x)

        return x


def get_initial_params(key):
    init_shape = jnp.ones((2), jnp.float32)
    initial_params = Net().init(key, init_shape)["params"]
    return initial_params


def create_optimizer(params, learning_rate):
    optimizer_def = optim.GradientDescent(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


def rmse_loss(sout, pred):
    return jnp.sqrt(jnp.mean((sout - pred) ** 2))


def compute_metrics(outputs, labels):
    rmse = rmse_loss(outputs, labels)
    metrics = {
        "rmse": rmse,
    }
    return metrics


def train_step(optimizer, batch, config):
    """Train for a single step."""
    var_layer = jnp.array([1] * (3 - 1) + [10])  # remove that from learn_pc?

    # stepping through every example indivdually
    outputs = []
    for x, y in zip(batch["data"], batch["label"]):
        grad, out = learn_pc(
            x,
            y,
            params=optimizer.target,
            act_fn=nn.tanh,
            beta=config.beta,
            it_max=config.inference_iterations,
            var_layer=var_layer,
        )
        optimizer = optimizer.apply_gradient(grad)
        outputs.append(out)
    outputs = jnp.stack(outputs)
    metrics = compute_metrics(outputs, batch["label"])
    return optimizer, metrics


def eval_step(params, batch):
    outputs = Net().apply({"params": params}, batch["data"])
    return compute_metrics(outputs, batch["label"])


def get_dataset():
    """Create the XOR data"""
    dataset = {
        "data": jnp.array(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        ).transpose(),
        "label": jnp.array([[1.0, 0.0, 0.0, 1.0]]).transpose(),
    }
    return dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.
    Args:
      config: Hyper-parameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The trained optimizer.
    """
    dataset = get_dataset()
    rng = jax.random.PRNGKey(config.random_seed)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    params = get_initial_params(init_rng)
    optimizer = create_optimizer(params, config.learning_rate)

    j_train_step = jax.jit(functools.partial(train_step, config=config))
    j_eval_step = jax.jit(eval_step)

    test_metrics = j_eval_step(optimizer.target, dataset)
    logging.info(
        "epoch: 0, train rmse: -.----, test rmse: %.4f ",
        test_metrics["rmse"],
    )
    summary_writer.scalar("RMSE Test", test_metrics["rmse"], 1)

    for epoch in range(2, config.num_epochs + 2):

        optimizer, train_metrics = j_train_step(optimizer, dataset)
        if (epoch + 1) % config.plotevery == 0:
            test_metrics = j_eval_step(optimizer.target, dataset)
            summary_writer.scalar("RMSE Test", test_metrics["rmse"], epoch)
            logging.info(
                "epoch: %d, train rmse: %.4f, test rmse: %.4f ",
                epoch + 1,
                train_metrics["rmse"],
                test_metrics["rmse"],
            )

        summary_writer.scalar("RMSE Train", train_metrics["rmse"], epoch)

    summary_writer.flush()
    return optimizer
