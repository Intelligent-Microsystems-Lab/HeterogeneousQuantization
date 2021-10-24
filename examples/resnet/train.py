# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""ImageNet example.

This script trains a ResNet-18 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
import resource

from absl import app
from absl import flags
from absl import logging
from clu import platform
from clu import metric_writers
from clu import periodic_actions

import models

from flax import jax_utils
from flax.training import common_utils

import jax
from jax import random

import ml_collections
from ml_collections import config_flags

import tensorflow as tf
import tensorflow_datasets as tfds


from train_utils import (
    TrainState,
    create_model,
    create_learning_rate_fn,
    create_train_state,
    restore_checkpoint,
    train_step,
    eval_step,
    sync_batch_stats,
    save_checkpoint,
    create_input_iter,
)
from load_pretrained_weights import load_pretrained_weights

# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


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

  image_size = 224

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
  train_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=True,
      cache=config.cache)
  eval_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=config.cache)

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
      model_cls=model_cls, config=config)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # Pre load weights.
  if config.pretrained and step_offset == 0:
    state = load_pretrained_weights(state, config.pretrained)

  # Reinitialize quant params.
  init_batch = next(train_iter)['image'][0, :, :, :, :]
  _, new_state = state.apply_fn({'params': state.params['params'],
                                 'quant_params': state.params['quant_params'],
                                 'batch_stats': state.batch_stats}, init_batch,
                                mutable=['batch_stats', 'quant_params',
                                         'weight_size', 'act_size'],)
  state = TrainState.create(apply_fn=state.apply_fn,
                            params={'params': state.params['params'],
                                    'quant_params': new_state['quant_params']},
                            tx=state.tx, batch_stats=state.batch_stats,
                            weight_size=state.weight_size,
                            act_size=state.act_size)

  state = jax_utils.replicate(state)
  # Debug note:
  # 1. Make above line a comment "state = jax_utils.replicate(state)".
  # 2. In train_util.py make all pmean commands comments.
  # 3. Use debug train_step.
  # 4. Swtich train and eval metrics lines.
  # 5. Uncomment JIT configs at the top.

  p_train_step = jax.pmap(
      functools.partial(train_step,
                        learning_rate_fn=learning_rate_fn,
                        weight_decay=config.weight_decay,
                        quant_target=config.quant_target),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  # # Debug
  # p_train_step = functools.partial(
  #     train_step,
  #     learning_rate_fn=learning_rate_fn,
  #     weight_decay=config.weight_decay,
  #     quant_target=config.quant_target)
  # p_eval_step = functools.partial(eval_step)

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)

    # # Debug
    # state, metrics = p_train_step(
    #     state, {'image': batch['image'][0, :, :, :],
    #             'label': batch['label'][0]})

    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        # # Debug
        # train_metrics = common_utils.stack_forest(train_metrics)
        summary = {
            f'{k}': v
            for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t)
        writer_train.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # # Debug
      # eval_metrics = common_utils.stack_forest(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      writer_eval.write_scalars(
          step + 1, {f'{key}': val for key, val in summary.items()})
      writer_eval.flush()
      writer_train.flush()
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
  tf.config.experimental.set_visible_devices([], 'TPU')

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
