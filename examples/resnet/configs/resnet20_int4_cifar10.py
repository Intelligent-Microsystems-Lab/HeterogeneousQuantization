# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial

from quant import uniform_static


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 1349194

  # As defined in the `models` module.
  config.model = 'ResNet20_CIFAR10'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'cifar10'
  config.num_classes = 10
  config.tfds_data_dir = None
  config.image_size = 32

  # Mean and std style for pre-processing.
  # config.mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  # config.stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

  # Edge models use inception-style MEAN & STDDEV for better post-quantization.
  config.mean_rgb = [127.0, 127.0, 127.0]
  config.stddev_rgb = [128.0, 128.0, 128.0]

  config.optimizer = 'sgd'
  config.learning_rate = .002
  config.lr_boundaries_scale = None  # {'80': .1, '120': .1}
  config.warmup_epochs = 20.
  config.momentum = 0.9
  config.batch_size = 128
  config.weight_decay = 0.0002
  config.nesterov = False
  config.smoothing = .0

  config.num_epochs = 160.0
  config.log_every_steps = 390

  config.cache = True
  config.half_precision = False

  config.pretrained = '../../pretrained_resnet/resnet20_cifar10'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()
  config.quant_target.size_div = 8. * 1024.

  config.quant = ml_collections.ConfigDict()

  config.quant.w_bits = 4
  config.quant.a_bits = 4

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(uniform_static)
  # no input quant in MixedDNN paper.
  # config.quant.stem.act = partial(uniform_static, act=True)

  config.quant.post_init = partial(
      uniform_static, act=True)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(uniform_static)
  #config.quant.mbconv.act = partial(uniform_static, act=True, init_bits = 6)
  config.quant.mbconv.nonl = partial(
      uniform_static, act=True)

  # Average quant.
  # config.quant.average = partial(uniform_static, act=True, init_bits = 6)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(uniform_static)
  # config.quant.dense.act = partial(uniform_static, act=True)
  config.quant.dense.bias = partial(uniform_static)

  return config
