# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial

from quant import parametric_d, double_mean_init


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 1349194

  # As defined in the `models` module.
  config.model = 'ResNet18'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012'
  config.tfds_data_dir = 'gs://imagenet_clemens/tensorflow_datasets'

  config.learning_rate = 0.0001
  config.warmup_epochs = 0.0
  config.momentum = 0.9
  config.batch_size = 2024
  config.weight_decay = 10e-4

  config.num_epochs = 1.0
  config.log_every_steps = 200

  config.cache = True
  config.half_precision = False

  config.pretrained = '../../../pretrained_resnet/resnet18'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = 8

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)
  config.quant.stem.act = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)
  config.quant.mbconv.act = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)

  # Average quant.
  config.quant.average = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)
  config.quant.dense.act = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)
  config.quant.dense.bias = partial(
      parametric_d, init_fn=double_mean_init, clip_quant_grads=False)

  return config
