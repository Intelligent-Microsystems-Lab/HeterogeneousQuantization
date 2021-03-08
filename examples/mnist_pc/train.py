"""MNIST example.
Library file which executes the training and evaluation loop for XOR.
"""
import functools
import time
from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
from flax.training import common_utils
from typing import Any
import jax
import jax.numpy as jnp
import ml_collections
from flax.linen.initializers import variance_scaling

from torchvision import datasets, transforms
import torch

import sys

sys.path.append("../..")
from predictive_coding import learn_pc  # noqa: E402

Array = Any

init_method = variance_scaling(1.0, "fan_avg", "uniform")


class Net(nn.Module):
    """A simple model."""

    @nn.compact
    def __call__(self, x: Array, act_fn: Any) -> Array:
        """Description of a forward pass
        Args:
            x: an array (inputs)
            act_fn: activation function applied after MVM
        Returns:
            An array containing the result.
        """
        x = act_fn(x)
        x = nn.Dense(features=600, kernel_init=init_method)(x)

        x = act_fn(x)
        x = nn.Dense(features=600, kernel_init=init_method)(x)

        x = act_fn(x)
        x = nn.Dense(features=10, kernel_init=init_method)(x)

        return x


def onehot(labels, num_classes=10):
    x = labels[..., None] == jnp.arange(num_classes)[None]
    x = jnp.where(x == 1, 0.97, 0.03)
    return x.astype(jnp.float32)


def get_initial_params(key, act_fn):
    init_shape = jnp.ones((784), jnp.float32)
    initial_params = Net().init(key, init_shape, act_fn)["params"]
    return initial_params


def create_optimizer(params, learning_rate):
    optimizer_def = optim.Adam(learning_rate=learning_rate)
    optimizer = optimizer_def.create(params)
    return optimizer


def rmse_loss(sout, pred):
    return jnp.sqrt(jnp.mean((sout - pred) ** 2))


def cross_entropy_loss(logits, labels):
    return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(outputs, labels):
    rmse = rmse_loss(outputs, onehot(labels))
    accuracy = jnp.mean(jnp.argmax(outputs, -1) == labels)
    metrics = {
        "loss": rmse,
        "accuracy": accuracy,
    }
    return metrics


def train_step(optimizer, batch, config):
    """Train for a single step."""
    var_layer = jnp.array([1] * (4 - 1) + [config.sigma_0])
    act_fn = string_to_act_fn(config.act_fn)
    v_learn_pc = jax.vmap(
        functools.partial(
            learn_pc,
            params=optimizer.target,
            act_fn=act_fn,
            beta=config.beta,
            it_max=config.inference_iterations,
            var_layer=var_layer,
        ),
        in_axes=(0, 0),
        out_axes=(0, 0),
    )

    grad, outputs = v_learn_pc(
        batch["image"],
        onehot(batch["labels"]),
    )
    grad = jax.tree_map(lambda x: x.mean(axis=0), grad)
    optimizer = optimizer.apply_gradient(grad)

    metrics = compute_metrics(outputs, batch["labels"])
    return optimizer, metrics


def eval_step(params, batch, act_fn):
    outputs = Net().apply({"params": params}, batch["image"], act_fn=act_fn)
    return compute_metrics(outputs, batch["labels"])


def inverse_logistic(x):
    return torch.log(1 / (1 - x))


def get_datasets(data_dir, batch_size):
    """Create the XOR data"""
    transform = transforms.Compose(
        [transforms.ToTensor(), torch.flatten, inverse_logistic]
    )
    dataset_train = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    dataset_eval = datasets.MNIST(data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset_eval, batch_size=batch_size
    )
    return train_loader, eval_loader


def string_to_act_fn(conf_str):
    if conf_str == "sigmoid":
        return nn.sigmoid
    if conf_str == "tanh":
        return nn.tanh
    else:
        raise Exception(conf_str + " is not a valid activation function")


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Execute model training and evaluation loop.
    Args:
      config: Hyper-parameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.
    Returns:
      The trained optimizer.
    """
    train_ds, eval_ds = get_datasets(config.data_dir, config.batch_size)
    steps_per_epoch = len(train_ds)
    num_steps = int(steps_per_epoch * config.num_epochs)
    act_fn = string_to_act_fn(config.act_fn)
    rng = jax.random.PRNGKey(config.random_seed)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    params = get_initial_params(init_rng, act_fn)
    optimizer = create_optimizer(params, config.learning_rate)

    j_train_step = jax.jit(functools.partial(train_step, config=config))
    j_eval_step = jax.jit(functools.partial(eval_step, act_fn=act_fn))

    epoch_metrics = []
    t_loop_start = time.time()
    for step in range(num_steps):
        batch = next(iter(train_ds))
        batch = {"image": jnp.array(batch[0]), "labels": jnp.array(batch[1])}
        optimizer, train_metrics = j_train_step(optimizer, batch)
        epoch_metrics.append(train_metrics)

        if (step + 1) % steps_per_epoch == 0:
            epoch_metrics = common_utils.stack_forest(epoch_metrics)
            summary_train = jax.tree_map(lambda x: x.mean(), epoch_metrics)

            for key, vals in epoch_metrics.items():  # type: ignore
                tag = "train_%s" % key
                for i, val in enumerate(vals):
                    summary_writer.scalar(tag, val, step - len(vals) + i + 1)
            steps_per_sec = steps_per_epoch / (time.time() - t_loop_start)
            t_loop_start = time.time()
            summary_writer.scalar("steps per second", steps_per_sec, step)

            epoch_metrics = []
            eval_metrics = []

            for batch in eval_ds:
                batch = {
                    "image": jnp.array(batch[0]),
                    "labels": jnp.array(batch[1]),
                }
                eval_metrics.append(j_eval_step(optimizer.target, batch))
            eval_metrics = common_utils.stack_forest(eval_metrics)
            summary_eval = jax.tree_map(lambda x: x.mean(), eval_metrics)

            for key, val in eval_metrics.items():  # type: ignore
                tag = "eval_%s" % key
                summary_writer.scalar(tag, val.mean(), step)
            summary_writer.flush()

            logging.info(
                "epoch: %d, train loss: %.4f, train acc: %.4f, eval loss:"
                " %.4f, eval acc: %.4f ",
                (step + 1) / steps_per_epoch,
                summary_train["loss"],
                summary_train["accuracy"],
                summary_eval["loss"],
                summary_eval["accuracy"],
            )

    return optimizer
