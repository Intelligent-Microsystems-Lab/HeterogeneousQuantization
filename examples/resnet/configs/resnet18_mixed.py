# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial

from quant import parametric_d_xmax


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 1349194

  # As defined in the `models` module.
  config.model = 'ResNet18'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012'
  config.num_classes = 1000
  config.tfds_data_dir = 'gs://imagenet_clemens/tensorflow_datasets'
  config.image_size = 224
  config.crop_padding = 32

  # Mean and std style for pre-processing.
  # config.mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  # config.stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

  # Edge models use inception-style MEAN & STDDEV for better post-quantization.
  config.mean_rgb = [127.0, 127.0, 127.0]
  config.stddev_rgb = [128.0, 128.0, 128.0]

  config.optimizer = 'sgd'
  config.learning_rate = 0.0002
  config.lr_boundaries_scale = None
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 2048
  config.weight_decay = 0.0001
  config.nesterov = True
  config.smoothing = .0

  config.num_epochs = 50.0
  config.log_every_steps = 100

  config.cache = True
  config.half_precision = False

  config.pretrained = '../../pretrained_resnet/resnet18_v2'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()

  config.quant_target.weight_mb = 5401
  config.quant_target.weight_penalty = .005
  config.quant_target.act_mode = 'max'
  config.quant_target.act_mb = 381
  config.quant_target.act_penalty = .005
  config.quant_target.size_div = 8. * 1000.
  config.quant_target.eval_start = 31050
  config.quant_target.update_every = 1

  config.quant = ml_collections.ConfigDict()

  config.quant.w_bits = 4
  config.quant.a_bits = 6

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(parametric_d_xmax)
  # no input quant in MixedDNN paper.
  # config.quant.stem.act = partial(parametric_d_xmax, act=True)

  config.quant.post_init = partial(
      parametric_d_xmax, act=True, bitwidth_min=1, xmax_max=255)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(parametric_d_xmax)
  #config.quant.mbconv.act = partial(parametric_d_xmax, act=True, init_bits = 6)
  config.quant.mbconv.nonl = partial(
      parametric_d_xmax, act=True, bitwidth_min=1, xmax_max=255)

  # Average quant.
  # config.quant.average = partial(parametric_d_xmax, act=True, init_bits = 6)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(parametric_d_xmax)
  # config.quant.dense.act = partial(parametric_d_xmax, act=True)
  config.quant.dense.bias = partial(parametric_d_xmax)

  return config
