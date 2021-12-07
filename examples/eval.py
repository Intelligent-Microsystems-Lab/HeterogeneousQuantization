# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

import functools
import resource
import time

from absl import app
from absl import flags
from absl import logging
from clu import platform

from flax import jax_utils

import input_pipeline
import models

from flax.training import common_utils

import jax
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np

import ml_collections
from ml_collections import config_flags


from train_utils import (
    TrainState,
    create_model,
    create_train_state,
    restore_checkpoint,
    eval_step
)

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def evaluate(config: ml_collections.ConfigDict,
             workdir: str) -> TrainState:

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size (' + str(config.batch_size) + ') must be \
      divisible by the number of devices (' + str(jax.device_count()) + ').')
  local_batch_size = config.batch_size // jax.process_count()

  dataset_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
  dataset_builder.download_and_prepare()
  if 'cifar10' in config.dataset:
    eval_iter = input_pipeline.create_input_iter_cifar10(
        dataset_builder, local_batch_size, train=False, config=config)

  elif 'imagenet2012' in config.dataset:
    eval_iter = input_pipeline.create_input_iter(
        dataset_builder, local_batch_size, train=False, config=config)
  else:
    raise Exception('Unrecognized data set: ' + config.dataset)

  val_or_test = "validation" if "imagenet" in config.dataset else "test"

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        val_or_test].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes, config=config)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, lambda x: x)
  state = restore_checkpoint(state, workdir)

  state = jax_utils.replicate(state, devices=jax.devices(
  )[:config.num_devices] if type(config.num_devices) == int else jax.devices())

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          size_div=config.quant_target.size_div,
          smoothing=config.smoothing
      ),
      axis_name='batch',
      devices=jax.devices()[:config.num_devices
                            ] if type(config.num_devices) == int else
      jax.devices())

  logging.info('Initial compilation, this might take some minutes...')
  eval_metrics = []
  time_per_epoch = []

  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)
    train_metrics_last_t = time.time()
    metrics = p_eval_step(state, eval_batch)
    time_per_epoch.append(time.time() - train_metrics_last_t)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.get_metrics(eval_metrics)
  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  logging.info('eval loss: %.4f, accuracy: %.2f latency: %.4f±%.4f(ms)',
               summary['loss'],
               summary['accuracy'] * 100,
               np.mean(time_per_epoch[1:]) * 1000,
               np.std(time_per_epoch[1:]) * 1000)
  for key, val in summary.items():
    logging.info('%s: %.4f', key, val)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'proc_index: {jax.process_index()}, '
                                       f'proc_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
