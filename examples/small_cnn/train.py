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

import jax
import jax.numpy as jnp

import sys

from flax.training import common_utils
from flax import optim

import tensorflow as tf
import tensorflow_datasets as tfds


# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

sys.path.append("../..")
from pc_modular import ConvolutionalPC, DensePC, FlattenPC, PC_NN  # noqa: E402


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


class test_pc_nn(PC_NN):
  def setup(self):
    self.layers = [
        ConvolutionalPC(
            features=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            non_linearity=jax.nn.relu,
            infer_lr=self.config.infer_lr,
            config=self.config.quant,
        ),
        # MaxPoolPC(window_shape=(2, 2), strides=(2, 2)),

        ConvolutionalPC(
            features=64,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="VALID",
            non_linearity=jax.nn.relu,
            infer_lr=self.config.infer_lr,
            config=self.config.quant,
        ),
        # MaxPoolPC(window_shape=(2, 2), strides=(2, 2)),


        ConvolutionalPC(
            features=64,
            kernel_size=(3, 3),
            padding="VALID",
            non_linearity=jax.nn.relu,
            infer_lr=self.config.infer_lr,
            config=self.config.quant,
        ),

        FlattenPC(config=self.config),
        DensePC(
            features=64,
            non_linearity=jax.nn.relu,
            infer_lr=self.config.infer_lr,
            cfg=self.config.quant,
        ),
        DensePC(
            features=self.config.num_classes,
            non_linearity=None,
            infer_lr=self.config.infer_lr,
            cfg=self.config.quant,
        ),
    ]


def mse(logits, target):
  return jnp.sum((logits - target) ** 2)


def cross_entropy_loss(logits, targt):

  logits = jax.nn.log_softmax(logits, axis=-1)

  return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)

  accuracy = jnp.mean(
      jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1)
  )

  return {"loss": loss, "accuracy": accuracy}


@functools.partial(jax.jit, static_argnums=(2, 3, 7))
def train_step(step, optimizer, lr_fn, nn_fn, batch, state, rng, cfg):
  label = jax.nn.one_hot(batch[1], num_classes=cfg.num_classes)

  (grads, logits), state = nn_fn(
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
  ds = ds.batch(int(cfg.batch_size))
  return ds.prefetch(tf.data.experimental.AUTOTUNE)


def train_and_evaluate(cfg, workdir):
  cfg = FrozenConfigDict(cfg)
  writer_train = metric_writers.create_default_writer(
      logdir=workdir + "/train", just_logging=jax.process_index() != 0
  )
  writer_eval = metric_writers.create_default_writer(
      logdir=workdir + "/eval", just_logging=jax.process_index() != 0
  )

  writer_train.write_hparams(
      {k: v for k, v in cfg.items() if k not in ["quant"]}
  )
  writer_train.write_hparams(cfg.quant)

  nn_cifar10 = test_pc_nn(config=cfg, loss_fn=cross_entropy_loss)
  nn_fn = nn_cifar10.apply

  # get data set
  ds_train = get_ds("train", cfg)
  ds_test = get_ds("test", cfg)

  rng = jax.random.PRNGKey(cfg.seed)
  rng, p_rng, subkey = jax.random.split(rng, 3)

  variables = nn_cifar10.init(
      p_rng,
      jnp.ones((int(cfg.batch_size), cfg.ds_xdim,
               cfg.ds_ydim, cfg.ds_channels)),
      subkey,
  )
  state, params = variables.pop("params")

  optimizer = optim.Adam(learning_rate=cfg.learning_rate).create(params)
  learning_rate_fn = (None,)

  for step in range(int(cfg.num_epochs)):
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

    writer_eval.write_scalars(step + 1, eval_metrics)
    writer_train.write_scalars(step + 1, train_metrics)

    writer_eval.flush()
    writer_train.flush()


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
