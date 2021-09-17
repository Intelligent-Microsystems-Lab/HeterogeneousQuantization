# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import platform

from ml_collections import config_flags, FrozenConfigDict

import functools
import time
import datetime
import json

import jax
import jax.numpy as jnp

import sys

from flax.metrics import tensorboard
from flax import linen as nn
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
from flax_qdense import QuantDense  # noqa: E402
from flax_qconv import QuantConv  # noqa: E402


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def dropout(x, rate, rng):
  mask = jax.random.bernoulli(rng, p=rate, shape=x.shape)
  return jax.lax.select(mask, x / rate, jnp.zeros_like(x))

class LeNet_BP(nn.Module):
  # LeNet with ReLU activations
  config: dict = None

  @nn.compact
  def __call__(self, x, rng):

    # rng, subkey = jax.random.split(rng, 2)
    # x = QuantConv(
    #     features=6,
    #     kernel_size=(5, 5),
    #     padding="VALID",
    #     use_bias=False,
    #     config=self.config,
    # )(x, subkey)
    # x = nn.relu(x)
    # # x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    # rng, subkey = jax.random.split(rng, 2)
    # x = QuantConv(
    #     features=16,
    #     kernel_size=(5, 5),
    #     padding="VALID",
    #     use_bias=False,
    #     config=self.config,
    # )(x, subkey)
    # x = nn.relu(x)
    # x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=512,
        use_bias=False,
        config=self.config,
    )(x, subkey)
    x = nn.relu(x)

    rng, subkey = jax.random.split(rng, 2)
    x = dropout(x, .2, subkey)

    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=128,
        use_bias=False,
        config=self.config,
    )(x, subkey)
    x = nn.relu(x)

    rng, subkey = jax.random.split(rng, 2)
    x = dropout(x, .2, subkey)

    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=self.config.num_classes,
        use_bias=False,
        config=self.config,
    )(x, subkey)
    return x


# def cross_entropy_loss(logits, targt):
#   targt = targt * (1.0 - cfg.label_smoothing) + (
#       cfg.label_smoothing / cfg.num_classes
#   )

#   logits = jax.nn.log_softmax(logits, axis=-1)

#   return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def mse(logits, target):
  return jnp.sum((logits - target) ** 2)


def compute_metrics(logits, labels):
  loss = mse(logits, labels)

  accuracy = jnp.mean(
      jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)
  )

  return {"loss": loss, "accuracy": accuracy}


@functools.partial(jax.jit, static_argnums=(2, 3, 7))
def train_step(step, optimizer, lr_fn, nn_fn, batch, state, rng, cfg):
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)

  def loss_fn(params):
    logits, _ = nn_fn(
        {"params": params, **state},
        batch[0],
        rng,
        mutable=list(state.keys()),
    )
    loss = mse(logits, label)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(optimizer.target)

  grads = jax.tree_map(lambda x: jnp.clip(x, -50, 50), grads)
  # lr = lr_fn(step)
  optimizer = optimizer.apply_gradient(grads)

  metrics = compute_metrics(logits, label)

  return optimizer, metrics


@functools.partial(jax.jit, static_argnums=(4, 5))
def eval_model(params, batch, state, rng, nn_fn, cfg):
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)

  out, _ = nn_fn(
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


def get_ds(split, cfg):
  (ds,), ds_info = tfds.load(
      cfg.ds,
      split=[split],
      shuffle_files=True,
      as_supervised=True,
      with_info=True,
  )
  ds = ds.map(
      normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
  )
  ds = ds.cache()
  ds = ds.shuffle(ds_info.splits[split].num_examples)
  ds = ds.batch(cfg.batch_size)
  return ds.prefetch(tf.data.experimental.AUTOTUNE)


def train_and_evaluate(cfg, workdir):
  # summary_writer = tensorboard.SummaryWriter(work_dir)
  # summary_writer.hparams(cfg)
  cfg = FrozenConfigDict(cfg)
  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )
  writer.write_hparams(cfg)

  nn_cifar10 = LeNet_BP(config=cfg)
  nn_fn = nn_cifar10.apply

  # with open(cfg.work_dir+'/config.json', 'w') as f:
  #  json.dump(cfg, f)

  # get data set
  ds_train = get_ds("train", cfg)
  ds_test = get_ds("test", cfg)

  rng = jax.random.PRNGKey(cfg.seed)
  rng, p_rng, subkey = jax.random.split(rng, 3)

  variables = nn_cifar10.init(
      p_rng, jnp.ones((cfg.batch_size, cfg.ds_xdim, cfg.ds_ydim, cfg.ds_channels)), subkey
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
  # logging.info(cfg)
  # logging.info(jax.devices())

  for step in range(cfg.num_epochs):
    # Do a batch of SGD.
    train_metrics = []
    for batch in tfds.as_numpy(ds_train):
      rng, subkey = jax.random.split(rng, 2)
      t_start = time.time()
      optimizer, metrics = train_step(
          step,
          optimizer,
          learning_rate_fn,
          nn_fn,
          batch,
          state,
          subkey,
          cfg,
      )
      metrics["accuracy"] = float(metrics["accuracy"])
      metrics["loss"] = float(metrics["loss"])
      metrics["time"] = time.time() - t_start
      train_metrics.append(metrics)

    eval_metrics = []
    for batch in tfds.as_numpy(ds_test):
      rng, subkey = jax.random.split(rng, 2)
      metrics = eval_model(
          optimizer.target, batch, state, subkey, nn_fn, cfg
      )
      eval_metrics.append(metrics)

    eval_metrics = common_utils.stack_forest(eval_metrics)
    eval_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

    train_metrics = common_utils.stack_forest(train_metrics)
    train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

    writer.write_scalars(
        step + 1, {f"eval_{key}": val for key, val in eval_metrics.items()}
    )
    writer.write_scalars(step + 1, train_metrics)
    # logging.info(
    #    "step: %d, train_loss: %.4f, train_accuracy: %.4f, "
    #    "test_loss: %.4f, test_accuracy: %.4f, time train batch: %.4f s",
    #    (step + 1),
    #    train_metrics["loss"],
    #    train_metrics["accuracy"],
    #    eval_metrics["loss"],
    #    eval_metrics["accuracy"],
    #    train_metrics["time"],
    # )
    writer.flush()


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  logging.info(
      "JAX process: %d / %d", jax.process_index(), jax.process_count()
  )
  logging.info("JAX local devices: %r", jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f"process_index: {jax.process_index()}, "
      f"process_count: {jax.process_count()}"
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, "workdir"
  )

  train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == "__main__":
  flags.mark_flags_as_required(["config", "workdir"])
  app.run(main)
