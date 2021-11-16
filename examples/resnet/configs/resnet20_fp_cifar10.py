# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial

from quant import parametric_d_xmax, double_mean_init


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 1349194

  # As defined in the `models` module.
  config.model = 'ResNet20_CIFAR10'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'cifar10'
  config.num_classes = 10
  config.tfds_data_dir = 'gs://imagenet_clemens/tensorflow_datasets'

  config.optimizer = 'sgd'
  config.learning_rate = .1
  config.lr_boundaries_scale = None  #{'80': .1, '120': .1}
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 1024
  config.weight_decay = 0.0001
  config.nesterov = False

  config.num_epochs = 160.
  config.log_every_steps = 100

  config.cache = True
  config.half_precision = False

  config.pretrained = None

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()
  config.quant_target.size_div = 8. * 1024.

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = 32

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()

  return config
