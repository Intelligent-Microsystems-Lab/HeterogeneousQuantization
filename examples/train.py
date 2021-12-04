# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Training script.

The data is loaded using tensorflow_datasets.
"""


import functools
import resource
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
import numpy as np

from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds


import ml_collections
from ml_collections import config_flags


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
)


# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

# import os
# os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
# os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
# os.environ["CLOUD_TPU_TASK_ID"] = "0"
# os.environ["TPU_VISIBLE_DEVICES"] = "0"

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

  # logging.get_absl_handler().use_absl_log_file('absl_logging', FLAGS.workdir)
  # logging.info('Git commit: ' + subprocess.check_output(
  #     ['git', 'rev-parse', 'HEAD']).decode('ascii').strip())
  # logging.info(config)

  rng = random.PRNGKey(config.seed)

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size (' + str(config.batch_size) + ') must be \
      divisible by the number of devices (' + str(jax.device_count()) + ').')

  local_batch_size = config.batch_size // jax.process_count()

  dataset_builder = tfds.builder(config.dataset, data_dir=config.tfds_data_dir)
  dataset_builder.download_and_prepare()
  if 'cifar10' in config.dataset:
    train_iter = input_pipeline.create_input_iter_cifar10(
        dataset_builder, local_batch_size, train=True, config=config)
    eval_iter = input_pipeline.create_input_iter_cifar10(
        dataset_builder, local_batch_size, train=False, config=config)

  elif 'imagenet2012' in config.dataset:
    train_iter = input_pipeline.create_input_iter(
        dataset_builder, local_batch_size, train=True, config=config)
    eval_iter = input_pipeline.create_input_iter(
        dataset_builder, local_batch_size, train=False, config=config)
  else:
    raise Exception('Unrecognized data set: ' + config.dataset)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  val_or_test = "validation" if "imagenet" in config.dataset else "test"

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        val_or_test].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10

  base_learning_rate = config.learning_rate * config.batch_size / 256.

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, num_classes=config.num_classes, config=config)

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # Pre load weights.
  if config.pretrained and step_offset == 0:
    state = model.load_model_fn(state, config.pretrained)

  # Reinitialize quant params.
  # TODO: @clee change this to train train_iter
  init_batch = next(eval_iter)['image'][0, :, :, :, :]
  rng, rng1, rng2 = jax.random.split(rng, 3)
  _, new_state = state.apply_fn({'params': state.params['params'],
                                 'quant_params': state.params['quant_params'],
                                 'batch_stats': state.batch_stats}, init_batch,
                                rng=rng1,
                                mutable=['batch_stats', 'quant_params',
                                         'weight_size', 'act_size'],
                                rngs={'dropout': rng2})
  state = TrainState.create(apply_fn=state.apply_fn,
                            params={'params': state.params['params'],
                                    'quant_params': new_state['quant_params']},
                            tx=state.tx, batch_stats=state.batch_stats,
                            weight_size=state.weight_size,
                            act_size=state.act_size)

  if len(state.weight_size) != 0:
    logging.info('Initial Network Weight Size in kB: ' + str(jnp.sum(jnp.array(
        jax.tree_util.tree_flatten(state.weight_size
                                   )[0])) / config.quant_target.size_div
    ) + ' init bits ' + str(config.quant.bits) + ' (No. Params: ' + str(
        jnp.sum(jnp.array(jax.tree_util.tree_flatten(state.weight_size)[0])
                ) / config.quant.bits) + ')')
  if len(state.act_size) != 0:
    logging.info('Initial Network Activation (Sum) Size in kB: ' + str(jnp.sum(jnp.array(jax.tree_util.tree_flatten(state.act_size)[0])) / config.quant_target.size_div
    ) + ' init bits ' + str(config.quant.bits) + ' (No. Params: ' + str(
        jnp.sum(jnp.array(
            jax.tree_util.tree_flatten(state.act_size
                                       )[0])) / config.quant.bits) + ')')
    logging.info('Initial Network Activation (Max) Size in kB: ' + str(jnp.max(jnp.array(jax.tree_util.tree_flatten(state.act_size)[0])) / config.quant_target.size_div
    ) + ' init bits ' + str(config.quant.bits) + ' (No. Params: ' + str(
        jnp.max(jnp.array(jax.tree_util.tree_flatten(state.act_size)[0])
                ) / config.quant.bits) + ')')


  # np.sum(jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: np.prod(x.shape), state.params['params']))[0])


  state = jax_utils.replicate(state, devices=jax.devices(
  )[:config.num_devices] if type(config.num_devices) == int else jax.devices())
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
          weight_decay=config.weight_decay,
          quant_target=config.quant_target,
          smoothing=config.smoothing
      ),
      axis_name='batch',
      devices=jax.devices()[:config.num_devices
                            ] if type(config.num_devices) == int else
      jax.devices())
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
  eval_metrics = []
  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)
    metrics = p_eval_step(state, eval_batch)
    eval_metrics.append(metrics)
  # eval_metrics = common_utils.get_metrics(eval_metrics)
  # Debug
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
  for step, batch in zip(range(step_offset, num_steps), train_iter):
    rng_list = jax.random.split(rng, (config.num_devices if type(
        config.num_devices) == int else jax.local_device_count()) + 1)
    rng = rng_list[0]

    
    
    state, metrics, grads = p_train_step(state, batch, rng_list[1:])
    # print(metrics['final_loss'][0])
    # import pdb; pdb.set_trace()
    # a = str(jnp.max(jnp.abs(state.params['params']['conv_init']['kernel'])))
    # b = str(jnp.max(jnp.abs(grads[0]['conv_init']['kernel'])))

    # c = str(state.params['quant_params']['conv_init']['parametric_d_xmax_0']['dynamic_range'][0])
    # d = str(state.params['quant_params']['conv_init']['parametric_d_xmax_0']['step_size'][0])

    # e = str(grads[1]['conv_init']['parametric_d_xmax_0']['dynamic_range'][0])
    # f = str(grads[1]['conv_init']['parametric_d_xmax_0']['step_size'][0])
    # print(a+','+b+','+c+','+d+','+e+','+f)
    #import pdb; pdb.set_trace()
    # print(str(metrics['ce_loss'][0]) + ',' + str(metrics['accuracy'].mean()) + ',' +str(metrics['size_weight_penalty'][0]*10))
    # print(
    #   str(state.params['params']['conv_init']['kernel'].max()) + ',' 
    #   + str( state.params['params']['conv_init']['kernel'].sum()) + ',' 
    #   + str( state.params['params']['conv_init']['kernel'].mean()) + ',' 
    #   + str(g_info[0][0]) + ',' 
    #   + str(g_info[1][0]) + ',' 
    #   + str(g_info[2][0])
    # )
    # except:
    #  import pdb; pdb.set_trace()
    # # Debug
    # state, metrics = p_train_step(
    #     state, {'image': batch['image'][0, :, :, :],
    #             'label': batch['label'][0]}, rng_list[2])

    #if metrics['decay'] > .2 and metrics['decay'] < 10:
    #  import pdb; pdb.set_trace()
    #print(metrics['decay'])




    # for h in hooks:
    #   h(step)
    # if step == step_offset:
    #   logging.info('Initial compilation completed.')

    # if config.get('log_every_steps'):
    #   train_metrics.append(metrics)
    #   if (step + 1) % config.log_every_steps == 0:
    #     train_metrics = common_utils.get_metrics(train_metrics)
    #     # # Debug
    #     # train_metrics = common_utils.stack_forest(train_metrics)
    #     summary = {
    #         f'{k}': v
    #         for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
    #     }
    #     summary['steps_per_second'] = config.log_every_steps / (
    #         time.time() - train_metrics_last_t)
    #     writer_train.write_scalars(step + 1, summary)
    #     writer_train.flush()
    #     train_metrics = []
    #     train_metrics_last_t = time.time()

    # if (step + 1) % steps_per_epoch == 0:
    #   epoch = (step + 1) // steps_per_epoch
    #   eval_metrics = []

    #   # sync batch statistics across replicas
    #   state = sync_batch_stats(state)
    #   for _ in range(steps_per_eval):
    #     eval_batch = next(eval_iter)
    #     metrics = p_eval_step(state, eval_batch)
    #     eval_metrics.append(metrics)
    #   eval_metrics = common_utils.get_metrics(eval_metrics)
    #   # # Debug
    #   # eval_metrics = common_utils.stack_forest(eval_metrics)
    #   summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
    #   logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
    #                epoch, summary['loss'], summary['accuracy'] * 100)
    #   writer_eval.write_scalars(
    #       step + 1, {f'{key}': val for key, val in summary.items()})
    #   writer_eval.flush()
    # if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
    #   state = sync_batch_stats(state)
    #   save_checkpoint(state, workdir)

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
