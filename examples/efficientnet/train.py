# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""EfficientNet example.

This script trains a EfficientB0 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""


import functools
import resource
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
from clu import platform
from clu import metric_writers
from clu import periodic_actions

from flax import jax_utils

import input_pipeline
import models

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state

import jax
from jax import lax
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds

import jax.numpy as jnp

import ml_collections
from ml_collections import config_flags

import optax


low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def create_model(*, model_cls, num_classes, **kwargs):
  model_dtype = jnp.float32
  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)

  @jax.jit
  def init(*args):
    return model.init(*args)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables['params'], variables['batch_stats']


def cross_entropy_loss(logits, labels, num_classes):
  one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels, num_classes):
  loss = cross_entropy_loss(logits, labels, num_classes)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def train_step(state, batch, learning_rate_fn, num_classes):
  """Perform a single training step."""
  def loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'],
        mutable=['batch_stats'])
    loss = cross_entropy_loss(logits, batch['label'], num_classes)
    weight_penalty_params = jax.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'], num_classes)
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'])

  return new_state, metrics


def eval_step(state, batch, num_classes):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(
      variables, batch['image'], train=False, mutable=False)
  return compute_metrics(logits, batch['label'], num_classes)


class TrainState(train_state.TrainState):
  batch_stats: Any


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  """Create initial training state."""

  params, batch_stats = initialized(rng, image_size, model)
  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
  )
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats,)
  return state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  writer_train = metric_writers.create_default_writer(
      logdir=workdir + '/train', just_logging=jax.process_index() != 0)
  writer_eval = metric_writers.create_default_writer(
      logdir=workdir + '/eval', just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size ('+str(config.batch_size)+') must be divisible by the number of devices ('+str(jax.device_count())+').')
  local_batch_size = config.batch_size // jax.process_count()

  dataset_builder = tfds.builder(config.dataset)
  dataset_builder.download_and_prepare()
  train_iter = input_pipeline.create_input_iter(
      dataset_builder, local_batch_size, train=True, config=config)
  eval_iter = input_pipeline.create_input_iter(
      dataset_builder, local_batch_size, train=False, config=config)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        'validation'].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10

  base_learning_rate = config.learning_rate * config.batch_size / 256.

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)

  state = create_train_state(
      rng, config, model, config.image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          num_classes=config.num_classes),
      axis_name='batch')
  p_eval_step = jax.pmap(functools.partial(
      eval_step, num_classes=config.num_classes), axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        writer_train.write_scalars(step + 1, summary)
        writer_train.flush()
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      writer_eval.write_scalars(
          step + 1, summary)
      writer_eval.flush()
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

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

  train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
