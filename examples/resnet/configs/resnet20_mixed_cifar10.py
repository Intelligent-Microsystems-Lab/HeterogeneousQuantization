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

  config.learning_rate = .01  # 0.0001
  config.lr_boundaries_scale = {'80': .1, '120': .1}
  config.warmup_epochs = 2.0
  config.momentum = 0.9
  config.batch_size = 1024  # 2048
  config.weight_decay = 0.0001
  config.nesterov = False

  config.num_epochs = 160.0
  config.log_every_steps = 100

  config.cache = True
  config.half_precision = False

  config.pretrained = '../../../pretrained_resnet/resnet20_cifar10.h5'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()

  config.quant_target.weight_mb = 70.1
  config.quant_target.weight_penalty = .1
  config.quant_target.act_mode = 'sum'
  config.quant_target.act_mb = 92.1
  config.quant_target.act_penalty = .1
  config.quant_target.size_div = 8. * 1024.  # 8_000 # mb or kb

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = 4

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(
      parametric_d_xmax)
  # no input quant in MixedDNN paper.
  # config.quant.stem.act = partial(parametric_d_xmax, act=True)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(
      parametric_d_xmax)
  config.quant.mbconv.act = partial(parametric_d_xmax, act=True)

  # Average quant.
  config.quant.average = partial(parametric_d_xmax, act=True)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(
      parametric_d_xmax)
  # config.quant.dense.act = partial(parametric_d_xmax, act=True)
  config.quant.dense.bias = partial(parametric_d_xmax)

  return config
