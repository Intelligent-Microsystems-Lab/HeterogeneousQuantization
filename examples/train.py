# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Training script.

The data is loaded using tensorflow_datasets.
"""


import functools
import subprocess
import time

from absl import app
from absl import flags
from absl import logging
from clu import platform
from clu import metric_writers
from clu import periodic_actions

from flax import jax_utils

import input_pipeline
import models

from flax.training import common_utils

import jax
import jax.numpy as jnp

from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds


import ml_collections
from ml_collections import config_flags


from train_utils import (
    TrainState,
    create_model,
    create_learning_rate_fn,
    create_penalty_fn,
    create_train_state,
    restore_checkpoint,
    train_step,
    eval_step,
    sync_batch_stats,
    save_checkpoint,
)


# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

# import os
# os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
# os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
# os.environ["CLOUD_TPU_TASK_ID"] = "0"
# os.environ["TPU_VISIBLE_DEVICES"] = "0"

# import resource
# low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

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

  logging.get_absl_handler().use_absl_log_file('absl_logging', FLAGS.workdir)
  logging.info('Git commit: ' + subprocess.check_output(
      ['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
  logging.info(config)

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size (' + str(config.batch_size) + ') must be \
      divisible by the number of devices (' + str(jax.device_count()) + ').')
  local_batch_size = config.batch_size // jax.process_count()

  dataset_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
  dataset_builder.download_and_prepare()
  if 'cifar10' in config.dataset:
    train_iter = input_pipeline.create_input_iter_cifar10(
        dataset_builder, local_batch_size, dtype=jnp.float32, train=True,
        cache=config.cache, mean_rgb=config.mean_rgb,
        std_rgb=config.stddev_rgb)
    eval_iter = input_pipeline.create_input_iter_cifar10(
        dataset_builder, local_batch_size, dtype=jnp.float32, train=False,
        cache=config.cache, mean_rgb=config.mean_rgb,
        std_rgb=config.stddev_rgb)
  elif 'imagenet2012' in config.dataset:
    train_iter = input_pipeline.create_input_iter(
        dataset_builder, local_batch_size, config.image_size,
        dtype=jnp.float32, train=True, cache=config.cache,
        mean_rgb=config.mean_rgb, std_rgb=config.stddev_rgb,
        crop=config.crop_padding, augment_name=config.augment_name)
    eval_iter = input_pipeline.create_input_iter(
        dataset_builder, config.eval_batch_size, config.image_size,
        dtype=jnp.float32, train=False, cache=config.cache,
        mean_rgb=config.mean_rgb, std_rgb=config.stddev_rgb,
        crop=config.crop_padding, augment_name=config.augment_name)
  else:
    raise Exception('Unrecognized data set: ' + config.dataset)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
    reload_for_finetune = num_steps + 1
    if 'pretraining' in config:
      num_steps += int(steps_per_epoch * config.pretraining.num_epochs)
      reload_for_finetune = num_steps + 1
    if 'finetune' in config:
      num_steps += int(steps_per_epoch * config.finetune.num_epochs)
  else:
    reload_for_finetune = config.num_train_steps + 1
    num_steps = config.num_train_steps

  val_or_test = "validation" if "imagenet" in config.dataset else "test"

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        val_or_test].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes, config=config)

  learning_rate_fn = create_learning_rate_fn(
      config, steps_per_epoch)

  decay_strength_fn = create_penalty_fn(
      config, steps_per_epoch)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, learning_rate_fn)

  # Pre load weights.
  if config.pretrained:
    state = model.load_model_fn(state, config.pretrained)

  # Reinitialize quant params.
  init_batch = next(train_iter)['image'][0, :, :, :, :]
  rng, rng1, rng2 = jax.random.split(rng, 3)
  _, new_state = state.apply_fn({'params': state.params['params'],
                                 'quant_params': state.params['quant_params'],
                                 'batch_stats': state.batch_stats,
                                 'quant_config': {}}, init_batch,
                                rng=rng1,
                                mutable=['batch_stats', 'quant_params',
                                         'weight_size', 'act_size',
                                         'quant_config'],
                                rngs={'dropout': rng2})
  state = TrainState.create(apply_fn=state.apply_fn,
                            params={'params': state.params['params'],
                                    'quant_params': new_state['quant_params']},
                            tx=state.tx, batch_stats=state.batch_stats,
                            weight_size=state.weight_size,
                            act_size=state.act_size,
                            quant_config=new_state['quant_config'])

  if len(state.weight_size) != 0:
    logging.info('Initial Network Weight Size in kB: ' + str(jnp.sum(jnp.array(
        jax.tree_util.tree_flatten(state.weight_size
                                   )[0])) / config.quant_target.size_div
    ) + ' init bits ' + str(config.quant.w_bits if 'w_bits' in config.quant
                            else config.quant.bits) + ' (No. Params: ' + str(
        jnp.sum(jnp.array(jax.tree_util.tree_flatten(state.weight_size)[0])
                ) / (config.quant.w_bits if 'w_bits' in config.quant
                     else config.quant.bits)) + ')')
  if len(state.act_size) != 0:
    logging.info('Initial Network Activation (Sum) Size in kB: ' + str(
        jnp.sum(jnp.array(jax.tree_util.tree_flatten(state.act_size)[0])
                ) / config.quant_target.size_div) + ' init bits ' + str(
        config.quant.a_bits if 'a_bits' in config.quant else
        config.quant.bits) + ' (No. Params: ' + str(jnp.sum(jnp.array(
            jax.tree_util.tree_flatten(state.act_size)[0])
        ) / (config.quant.a_bits if 'a_bits' in config.quant else
             config.quant.bits)) + ')')
    logging.info('Initial Network Activation (Max) Size in kB: ' + str(
        jnp.max(jnp.array(jax.tree_util.tree_flatten(state.act_size)[0])
                ) / config.quant_target.size_div) + ' init bits ' + str(
        config.quant.a_bits if 'a_bits' in config.quant else
        config.quant.bits) + ' (No. Params: ' + str(jnp.max(jnp.array(
            jax.tree_util.tree_flatten(state.act_size)[0])
        ) / (config.quant.a_bits if 'a_bits' in config.quant else
             config.quant.bits)) + ')')

  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  state = jax_utils.replicate(state)
  # Debug note:
  # 1. Make above line a comment "state = jax_utils.replicate(state)".
  # 2. In train_util.py make all pmean commands comments.
  # 3. Use debug train_step.
  # 4. Swtich train and eval metrics lines.
  # 5. Uncomment JIT configs at the top.

  p_train_step = jax.pmap(
      functools.partial(
          train_step,
          learning_rate_fn=learning_rate_fn,
          decay_strength_fn=decay_strength_fn,
          weight_decay=config.weight_decay,
          quant_target=config.quant_target,
          smoothing=config.smoothing,
      ),
      axis_name='batch',
  )

  p_eval_step = jax.pmap(
      functools.partial(
          eval_step,
          size_div=config.quant_target.size_div,
          smoothing=config.smoothing
      ),
      axis_name='batch',
  )

  # # Debug
  # p_train_step = functools.partial(
  #     train_step,
  #     learning_rate_fn=learning_rate_fn,
  #     weight_decay=config.weight_decay,
  #     quant_target=config.quant_target,
  #     smoothing=config.smoothing)
  # p_eval_step = functools.partial(
  #     eval_step,
  #     size_div=config.quant_target.size_div,
  #     smoothing=config.smoothing)

  # Initial Accurcay
  logging.info('Start evaluating model at beginning...')
  eval_metrics = []
  eval_best = -1.
  evaled_steps = 0
  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)
    metrics = p_eval_step(state, eval_batch)
    eval_metrics.append(metrics)
  eval_metrics = common_utils.stack_forest(eval_metrics)
  summary = jax.tree_map(lambda x: jnp.mean(x), eval_metrics)
  logging.info('Initial, loss: %.10f, accuracy: %.10f',
               summary['loss'], summary['accuracy'] * 100)

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in zip(range(int(step_offset),
                               int(num_steps)), train_iter):

    if step == reload_for_finetune:
      state = jax_utils.unreplicate(state)
      state = restore_checkpoint(state, workdir + '/best')
      state = jax_utils.replicate(state)
      logging.info('Starting finetuning, restored best checkpoint...')

    rng_list = jax.random.split(rng, jax.local_device_count() + 1)
    rng = rng_list[0]

    # alternating phases
    if (step + 1) % config.quant_target.update_every == 0:
      b_quant = jnp.ones((jax.local_device_count(),))
    else:
      b_quant = jnp.zeros((jax.local_device_count(),))

    if 'pretraining' in config:
      if step < config.pretraining.num_epochs * steps_per_epoch:
        b_quant = jnp.zeros((jax.local_device_count(),))

    if 'finetune' in config:
      if step > (config.num_epochs + config.pretraining.num_epochs
                 ) * steps_per_epoch:
        b_quant = jnp.zeros((jax.local_device_count(),))

    state, metrics = p_train_step(state, batch, rng_list[1:], b_quant)

    # # Debug
    # state, metrics = p_train_step(
    #     state, {'image': batch['image'][0, :, :, :] * 0 + 1,
    #             'label': batch['label'][0] * 0 + 1}, rng_list[2])

    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    # evalaute when constraints are fullfilled
    if 'act_mb' in config.quant_target and 'weight_mb' in config.quant_target:
      if (evaled_steps + 1) % 5 == 0:
        # step > config.quant_target.eval_start:
        # evaluate network size after gradients are applied.
        evaled_steps += 1
        metrics_size = p_eval_step(state, batch)
        weight_cond = (
            metrics_size['weight_size'].mean() <= config.quant_target.weight_mb
        )
        act_cond = (((config.quant_target.act_mode == 'max'
                      ) and (
            metrics_size['act_size_max'].mean() <= config.quant_target.act_mb)
        ) or ((config.quant_target.act_mode == 'sum') and (
            metrics_size['act_size_sum'].mean() <= config.quant_target.act_mb)
        ))

        if weight_cond and act_cond:
          # sync batch statistics across replicas
          eval_metrics = []
          state = sync_batch_stats(state)
          for _ in range(steps_per_eval):
            eval_batch = next(eval_iter)
            size_metrics = p_eval_step(state, eval_batch)
            eval_metrics.append(size_metrics)
          eval_metrics = common_utils.get_metrics(eval_metrics)
          # # Debug
          # eval_metrics = common_utils.stack_forest(eval_metrics)
          summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
          if summary['accuracy'] > eval_best:
            save_checkpoint(state, workdir + '/best')
            logging.info('!!! Saved new best model with accuracy %.4f weight'
                         'size %.4f max act %.4f sum act %.4f',
                         summary['accuracy'], summary['weight_size'],
                         summary['act_size_max'], summary['act_size_sum'])
            eval_best = summary['accuracy']

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
        writer_train.flush()
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      epoch = (step + 1) // steps_per_epoch
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
      logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      writer_eval.write_scalars(
          step + 1, {f'{key}': val for key, val in summary.items()})
      writer_eval.flush()
      if (('act_mb' not in config.quant_target
           ) and ('weight_mb' not in config.quant_target
                  ) and (summary['accuracy'] > eval_best)):
        save_checkpoint(state, workdir + '/best')
        logging.info('!!!! Saved new best model with accuracy %.4f',
                     summary['accuracy'])
        eval_best = summary['accuracy']

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
