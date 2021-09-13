# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags, FrozenConfigDict

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

import tensorflow as tf
import tensorflow_datasets as tfds


from configs.default import get_config

# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

sys.path.append("../..")
from pc_modular import ConvolutionalPC, DensePC, FlattenPC, PC_NN  # noqa: E402


tf.config.set_visible_devices([], 'GPU')
cfg = get_config()
cfg = FrozenConfigDict(cfg)


class test_pc_nn(PC_NN):
  def setup(self):
    self.layers = [
        ConvolutionalPC(
            features=6,
            kernel_size=(5, 5),
            padding="VALID",
            non_linearity=jax.nn.relu,
            config=cfg,

        ),
        #x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        ConvolutionalPC(
            features=16,
            kernel_size=(5, 5),
            padding="VALID",
            non_linearity=jax.nn.relu,
            config=cfg,

        ),
        # x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        FlattenPC(config=cfg),
        DensePC(features=200, non_linearity=jax.nn.relu, config=cfg, ),
        DensePC(features=150, non_linearity=jax.nn.relu, config=cfg, ),
        DensePC(features=10, non_linearity=None, config=cfg, ),
    ]


def mse(logits, target):
  return jnp.sum((logits - target) ** 2)


nn_cifar10 = test_pc_nn(config=cfg, loss_fn=mse)


# def cross_entropy_loss(logits, targt):
#   targt = targt * (1.0 - cfg.label_smoothing) + (
#       cfg.label_smoothing / cfg.num_classes
#   )

#   logits = jax.nn.log_softmax(logits, axis=-1)

#   return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = mse(logits, labels)

  accuracy = jnp.mean(
      jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)
  )

  return {"loss": float(loss), "accuracy": float(accuracy)}


# @functools.partial(jax.jit, static_argnums=(2))
def train_step(step, optimizer, lr_fn, batch, state, rng):
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)

  (grads, logits), state = nn_cifar10.apply(
      {"params": optimizer.target, **state},
      batch[0],
      label,
      rng,
      mutable=list(state.keys()),
      method=PC_NN.grads,
  )

  grads = jax.tree_map(lambda x: jnp.clip(x, -50, 50), grads)

  # lr = lr_fn(step)
  optimizer = optimizer.apply_gradient(grads)

  metrics = compute_metrics(logits, label)

  return optimizer, metrics


@jax.jit
def eval_model(params, batch, state, rng):
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)

  out, _ = nn_cifar10.apply(
      {"params": params, **state},
      batch[0],
      rng,
      mutable=list(state.keys()),
  )

  metrics = compute_metrics(out, label)
  return metrics


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.0, label


def get_ds(split):
  (ds_train, ds_test), ds_info = tfds.load(
      "cifar10",
      split=["train", "test"],
      shuffle_files=True,
      as_supervised=True,
      with_info=True,
  )

  ds_train = ds_train.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  ds_train = ds_train.cache()
  #ds_train = ds_train.shuffle(ds_info.splits[split].num_examples)
  ds_train = ds_train.batch(cfg.batch_size)
  return ds_train.prefetch(tf.data.experimental.AUTOTUNE)


def main(_):
  summary_writer = tensorboard.SummaryWriter(cfg.work_dir)
  summary_writer.hparams(cfg)

  # get data set
  ds_train = get_ds("train")
  ds_test = get_ds("test")

  rng = jax.random.PRNGKey(cfg.seed)
  rng, p_rng, subkey = jax.random.split(rng, 3)

  variables = nn_cifar10.init(
      p_rng, jnp.ones((cfg.batch_size, 32, 32, 3)), subkey
  )
  state, params = variables.pop("params")

  optimizer = optim.GradientDescent(learning_rate=cfg.learning_rate).create(
      params
  )
  learning_rate_fn = (None,)  # create_cosine_learning_rate_schedule(
  #    cfg.learning_rate,
  #    len(ds_train),
  #    cfg.num_epochs,
  #    warmup_length=cfg.warmup_epochs,
  # )

  # Training loop.
  logging.info(cfg)
  logging.info(jax.devices())

  for step in range(cfg.num_epochs):
    # Do a batch of SGD.
    train_metrics = []
    for batch in tfds.as_numpy(ds_train):
      rng, subkey = jax.random.split(rng, 2)
      t_start = time.time()
      optimizer, metrics = train_step(
          step, optimizer, learning_rate_fn, batch, state, subkey
      )
      metrics["time"] = time.time() - t_start
      train_metrics.append(metrics)

    eval_metrics = []
    for batch in tfds.as_numpy(ds_test):
      rng, subkey = jax.random.split(rng, 2)
      metrics = eval_model(optimizer.target, batch, state, subkey)
      eval_metrics.append(metrics)

    eval_metrics = common_utils.stack_forest(eval_metrics)
    eval_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

    train_metrics = common_utils.stack_forest(train_metrics)
    train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

    logging.info(
        "step: %d, train_loss: %.4f, train_accuracy: %.4f, "
        "test_loss: %.4f, test_accuracy: %.4f, time train batch: %.4f s",
        (step + 1),
        train_metrics["loss"],
        train_metrics["accuracy"],
        eval_metrics["loss"],
        eval_metrics["accuracy"],
        train_metrics["time"],
    )


if __name__ == "__main__":
  app.run(main)
