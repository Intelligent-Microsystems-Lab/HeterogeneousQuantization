# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Training script with distillation
The data is loaded using tensorflow_datasets.
For distillation we use EfficientNet-L2.
Parts copied from
https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
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
from jax import lax
import numpy as np

from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds


import ml_collections
from ml_collections import config_flags

import train_utils
import optax

from train_utils import (
    TrainState,
    create_model,
    create_learning_rate_fn,
    create_penalty_fn,
    create_train_state,
    restore_checkpoint,
    eval_step,
    sync_batch_stats,
    save_checkpoint,
)


import sys
sys.path.append('../../vision_transformer')
from vit_jax.configs import models as models_config  # noqa: E402
from vit_jax import models as models_vit  # noqa: E402
from vit_jax import checkpoint as checkpoint_vit  # noqa: E402


# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache, mean_rgb, std_rgb, crop, augment_name):
  ds = create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache, mean_rgb=mean_rgb, std_rgb=std_rgb, crop=crop,
      augment_name=augment_name)
  it = map(input_pipeline.prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


def create_split(dataset_builder, batch_size, train, dtype,
                 image_size, cache, mean_rgb, std_rgb, crop, augment_name):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.
  Args:
    dataset_builder: TFDS dataset builder for ImageNet.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    dtype: data type of the image.
    image_size: The target size of the images.
    cache: Whether to cache the dataset.
  Returns:
    A `tf.data.Dataset`.
  """
  if train:
    train_examples = dataset_builder.info.splits['train'].num_examples
    split_size = train_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'train[{}:{}]'.format(start, start + split_size)
  else:
    validate_examples = dataset_builder.info.splits['validation'].num_examples
    split_size = validate_examples // jax.process_count()
    start = jax.process_index() * split_size
    split = 'validation[{}:{}]'.format(start, start + split_size)

  def decode_example(example):
    if train:
      image = input_pipeline.preprocess_for_train(
          example['image'], dtype, image_size, mean_rgb, std_rgb, crop,
          augment_name)
      image2 = input_pipeline.preprocess_for_eval(
          example['image'], dtype, 384, mean_rgb, std_rgb, crop)
    else:
      image = input_pipeline.preprocess_for_eval(
          example['image'], dtype, image_size, mean_rgb, std_rgb, crop)
      image2 = input_pipeline.preprocess_for_eval(
          example['image'], dtype, 384, mean_rgb, std_rgb, crop)
    return {'image': image, 'image2': image2, 'label': example['label']}

  ds = dataset_builder.as_dataset(split=split, decoders={
      'image': tfds.decode.SkipDecoding(),
  })
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(10)

  return ds


def train_step(state, batch, rng, logits_tgt, learning_rate_fn,
               decay_strength_fn, weight_decay, quant_target,
               smoothing,):
  """Perform a single training step."""
  rng, prng = jax.random.split(rng, 2)
  step = state.step

  def loss_fn(params, inputs, targets, quant_params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn({'params': params,
                                              'quant_params': quant_params,
                                              'batch_stats': state.batch_stats,
                                              'weight_size': state.weight_size,
                                              'act_size': state.act_size,
                                              'quant_config':
                                              state.quant_config},
                                             inputs, rng=prng, mutable=[
        'batch_stats', 'weight_size', 'act_size',
        'quant_config'],
        rngs={'dropout': rng})

    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits, labels=jax.nn.softmax(logits_tgt, axis=-1)))
    return loss, (new_model_state, logits)

  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 3], has_aux=True)
  aux, grads = grad_fn(
      state.params['params'], batch['image'], batch['label'],
      state.params['quant_params'])

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  grads = (grads[0], jax.tree_util.tree_map(
      lambda x: x * 0., grads[1]))

  new_model_state, logits = aux[1]

  metrics = train_utils.compute_metrics(
      logits, batch['label'], new_model_state, quant_target.size_div,
      smoothing)
  metrics['learning_rate'] = lr
  new_state = state.apply_gradients(
      grads={'params': grads[0], 'quant_params': grads[1]},
      batch_stats=new_model_state['batch_stats'],
      weight_size=new_model_state['weight_size'],
      act_size=new_model_state['act_size'],
      quant_config=new_model_state['quant_config'])

  metrics['final_loss'] = aux[0]
  metrics['accuracy'] = metrics['accuracy']

  return new_state, metrics


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
  if 'imagenet2012' in config.dataset:
    train_iter = create_input_iter(
        dataset_builder, local_batch_size, config.image_size,
        dtype=jnp.float32, train=True, cache=config.cache,
        mean_rgb=config.mean_rgb, std_rgb=config.stddev_rgb,
        crop=config.crop_padding, augment_name=config.augment_name)
    eval_iter = create_input_iter(
        dataset_builder, local_batch_size, config.image_size,
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

  # restore from checkpoint.
  state = restore_checkpoint(state, config.restore_path)
  state = state.replace(step=0)

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
  # 6. Deactivate logging handler.

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
  #     decay_strength_fn=decay_strength_fn,
  #     weight_decay=config.weight_decay,
  #     quant_target=config.quant_target,
  #     smoothing=config.smoothing,)
  # p_eval_step = functools.partial(
  #     eval_step,
  #     size_div=config.quant_target.size_div,
  #     smoothing=config.smoothing)

  # Initial Accurcay
  logging.info('Start evaluating model at beginning...')
  eval_metrics = []
  eval_best = -1.

  # teacher model
  big_model = "../../pretrained_efficientnet/B_16-i21k-300ep-lr_0.001-aug_" \
      + "medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-" \
      + "res_384.npz"
  model_config = models_config.AUGREG_CONFIGS['B_16']
  model = models_vit.VisionTransformer(
      num_classes=1000, **model_config)
  params = checkpoint_vit.load(big_model)

  p_teacher_logits = jax.pmap(
      lambda x: model.apply({'params': params}, x, train=False),
      axis_name="batch",
  )
  eval_record = []

  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)
    logits = p_teacher_logits(eval_batch['image2'])
    eval_record.append(jnp.argmax(logits, axis=-1) == eval_batch['label'])

    metrics = p_eval_step(state, eval_batch)
    eval_metrics.append(metrics)

  logging.info('Teacher model accuracy %.5f', np.mean(eval_record) * 100)

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

    rng_list = jax.random.split(rng, jax.local_device_count() + 1)
    rng = rng_list[0]

    logits = p_teacher_logits(batch['image2'])
    state, metrics = p_train_step(state, batch, rng_list[1:], logits)

    # # Debug
    # state, metrics = p_train_step(
    #     state, {'image': batch['image'][0, :, :, :],
    #             'label': batch['label'][0]}, rng_list[2], logits[0,:,:])

    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.stack_forest(train_metrics)
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
      eval_metrics = common_utils.stack_forest(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)
      writer_eval.write_scalars(
          step + 1, {f'{key}': val for key, val in summary.items()})
      writer_eval.flush()
      if summary['accuracy'] > eval_best:
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
