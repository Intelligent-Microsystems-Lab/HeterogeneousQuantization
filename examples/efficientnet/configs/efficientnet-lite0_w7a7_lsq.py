# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial
from quant import parametric_d


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = 'EfficientNetB0'

  # `name` argument of tensorflow_datasets.builder()
  config.cache = True
  config.dataset = 'imagenet2012'
  config.tfds_data_dir = 'gs://imagenet_clemens/tensorflow_datasets'
  config.image_size = 224
  config.crop_padding = 32

  # Mean and std style for pre-processing.
  # config.mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  # config.stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

  # Edge models use inception-style MEAN & STDDEV for better post-quantization.
  config.mean_rgb = [127.0, 127.0, 127.0]
  config.stddev_rgb = [128.0, 128.0, 128.0]

  config.num_classes = 1000

  # Load pretrained weights.
  config.pretrained = "../../../pretrained_efficientnet/efficientnet-lite0"

  config.learning_rate = 0.0001
  config.warmup_epochs = 2  # for optimizer to settle in
  config.lr_boundaries = [16, 32]
  config.lr_scales = [1 / 10, 1 / 10]
  config.weight_decay = 1e-5
  config.momentum = 0.9
  config.batch_size = 2048

  config.num_epochs = 50
  config.log_every_steps = 256

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant = ml_collections.ConfigDict()

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(parametric_d, bits=7)
  config.quant.stem.act = partial(parametric_d, bits=7, act=True)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(parametric_d, bits=7)
  config.quant.mbconv.act = partial(parametric_d, bits=7, act=True)

  # Conv for head layer.
  config.quant.head = ml_collections.ConfigDict()
  config.quant.head.weight = partial(parametric_d, bits=7)
  config.quant.head.act = partial(parametric_d, bits=7, act=True)

  # Average quant.
  config.quant.average = partial(parametric_d, bits=7, act=True)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(parametric_d, bits=7)
  config.quant.dense.act = partial(parametric_d, bits=7, act=True)
  config.quant.dense.bias = partial(parametric_d, bits=7)

  return config
