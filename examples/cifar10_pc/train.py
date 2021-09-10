# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

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

import ml_collections

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

sys.path.append("../..")
from pc_modular import DensePC, PC_NN  # noqa: E402

cfg = ml_collections.ConfigDict()
cfg.seed = 203853699


cfg.work_dir = (
    "../../../training_dir/cifar10_pc-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    )
)
cfg.batch_size = 128
cfg.num_epochs = 10
cfg.warmup_epochs = 5
cfg.momentum = 0.9
cfg.learning_rate = 0.1
cfg.infer_lr = 0.2
cfg.infer_steps = 100
cfg.num_classes = 10
cfg.label_smoothing = 0.1


def cross_entropy_loss(logits, targt):
  targt = targt * (1.0 - cfg.label_smoothing) + (
      cfg.label_smoothing / cfg.num_classes
  )

  logits = jax.nn.log_softmax(logits, axis=-1)

  return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)

  accuracy = jnp.mean(jnp.argmax(logits.sum(0), axis=-1) == labels)

  return {"loss": loss, "accuracy": accuracy}


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(step, optimizer, lr_fn, batch, state):
  image = batch[0].reshape((-1, 3072))
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)
  grads, state = nn_cifar10.apply(
      {"params": optimizer.target, **state},
      image,
      label,
      mutable=list(state.keys()),
      method=PC_NN.grads,
  )

  lr = lr_fn(step)
  optimizer = optimizer.apply_gradient(grads, learning_rate=lr)

  metrics = compute_metrics(
      state["pc"][list(state["pc"].keys())[-1]]["out"], label
  )

  return optimizer, metrics


@jax.jit
def eval_model(params, batch, state):

  image = batch[0].reshape((-1, 3072))
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)

  out, state = nn_cifar10.apply(
      {"params": params, **state},
      train_x,
      mutable=list(state.keys()),
  )

  metrics = compute_metrics(out, label)

  return metrics


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.0, label


class test_pc_nn(PC_NN):
  def setup(self):
    self.layers = [
        DensePC(100, config=cfg),
        DensePC(100, config=cfg),
        DensePC(10, config=cfg),
    ]


nn_cifar10 = test_pc_nn(config=cfg, loss_fn=cross_entropy_loss)


def main(_):
  summary_writer = tensorboard.SummaryWriter(cfg.work_dir)
  summary_writer.hparams(cfg)

  # get data set
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
  ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
  ds_train = ds_train.batch(cfg.batch_size)
  ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

  ds_test = ds_test.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  ds_test = ds_test.batch(cfg.batch_size)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

  rng = jax.random.PRNGKey(cfg.seed)
  rng, p_rng = jax.random.split(rng, 2)

  variables = nn_cifar10.init(p_rng, jnp.ones((cfg.batch_size, 3072)))
  state, params = variables.pop("params")

  optimizer = optim.Momentum(beta=cfg.momentum, nesterov=True).create(params)
  learning_rate_fn = create_cosine_learning_rate_schedule(
      cfg.learning_rate,
      len(ds_train),
      cfg.num_epochs,
      warmup_length=cfg.warmup_epochs,
  )

  # Training loop.
  logging.info(cfg)
  logging.info(jax.devices())

  for step in range(cfg.num_epochs):
    # Do a batch of SGD.
    train_metrics = []
    for batch in tfds.as_numpy(ds_train):
      t_start = time.time()
      optimizer, metrics = train_step(
          step, optimizer, learning_rate_fn, batch, state
      )
      metrics["time"] = time.time() - t_start
      train_metrics.append(metrics)

    eval_metrics = []
    for image, label in tfds.as_numpy(ds_test):
      metrics = eval_model(optimizer.target, batch, state)
      eval_metrics.append(metrics)

    eval_metrics = common_utils.stack_forest(eval_metrics)
    eval_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

    train_metrics = common_utils.stack_forest(train_metrics)
    train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

    logging.info(
        "step: %d, train_loss: %.4f, train_accuracy: %.4f, "
        "test_loss: %.4f, test_accuracy: %.4f, ~length train batch: %.4f s",
        (step + 1),
        train_metrics["loss"],
        train_metrics["accuracy"],
        eval_metrics["loss"],
        eval_metrics["accuracy"],
    )


if __name__ == "__main__":
  app.run(main)
