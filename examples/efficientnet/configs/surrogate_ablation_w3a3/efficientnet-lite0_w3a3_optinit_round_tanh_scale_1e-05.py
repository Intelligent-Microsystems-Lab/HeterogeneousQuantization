# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial
from quant import uniform_static, gaussian_init, percentile_init, round_tanh


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
  config.learning_rate = 0.01
  config.lr_boundaries_scale = None
  config.warmup_epochs = 2.0
  config.momentum = 0.9
  config.batch_size = 2048
  config.weight_decay = 0.0001
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

  config.quant.g_scale = 1e-05

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(uniform_static, init_fn=partial(gaussian_init), round_fn = round_tanh)
  config.quant.stem.act = partial(uniform_static, init_fn=partial(percentile_init, perc=99.9), round_fn = round_tanh)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(uniform_static, init_fn=partial(gaussian_init), round_fn = round_tanh)
  config.quant.mbconv.act = partial(uniform_static, init_fn=partial(percentile_init, perc=99.9), round_fn = round_tanh)

  # Conv for head layer.
  config.quant.head = ml_collections.ConfigDict()
  config.quant.head.weight = partial(uniform_static, init_fn=partial(gaussian_init), round_fn = round_tanh)
  config.quant.head.act = partial(uniform_static, init_fn=partial(percentile_init, perc=99.9), round_fn = round_tanh)

  # Average quant.
  config.quant.average = partial(uniform_static, init_fn=partial(percentile_init, perc=99.9), round_fn = round_tanh)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(uniform_static, init_fn=partial(gaussian_init), round_fn = round_tanh)
  config.quant.dense.act = partial(uniform_static, init_fn=partial(percentile_init, perc=99.9), round_fn = round_tanh)
  config.quant.dense.bias = partial(uniform_static, init_fn=partial(gaussian_init), round_fn = round_tanh)

  return config
