# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial
from quant import uniform_static, percentile_init, gaussian_init, round_invtanh


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = 'EfficientNetB0'
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

  config.optimizer = 'rmsprop'
  config.learning_rate = 0.00125
  config.lr_boundaries_scale = None
  config.warmup_epochs = 2.0
  config.momentum = 0.9
  config.batch_size = 2048
  config.weight_decay = 0.00001
  config.nesterov = True
  config.smoothing = .1

  config.num_epochs = 50
  config.log_every_steps = 256

  config.cache = True

  # Load pretrained weights.
  config.pretrained = "../../pretrained_efficientnet/efficientnet-lite0"

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()
  config.quant_target.size_div = 8. * 1024.

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = 3

  config.quant.g_scale = 5e-06

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial, round_fn = round_invtanh)
      uniform_static, init_fn=partial(gaussian_init))

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial, round_fn = round_invtanh)
      uniform_static, init_fn=partial(gaussian_init))
  config.quant.mbconv.act = partial, round_fn = round_invtanh)
      uniform_static, act=True, init_fn=partial(percentile_init, perc=99.9))

  # Conv for head layer.
  config.quant.head = ml_collections.ConfigDict()
  config.quant.head.weight = partial, round_fn = round_invtanh)
      uniform_static, init_fn=partial(gaussian_init))
  config.quant.head.act = partial, round_fn = round_invtanh)
      uniform_static, act=True, init_fn=partial(percentile_init, perc=99.9))

  # Average quant.
  config.quant.average = partial, round_fn = round_invtanh)
      uniform_static, act=True, init_fn=partial(percentile_init, perc=99.9))

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial, round_fn = round_invtanh)
      uniform_static, init_fn=partial(gaussian_init))
  config.quant.dense.act = partial, round_fn = round_invtanh)
      uniform_static, act=True, init_fn=partial(percentile_init, perc=99.9))
  config.quant.dense.bias = partial, round_fn = round_invtanh)
      uniform_static, init_fn=partial(gaussian_init))

  return config
