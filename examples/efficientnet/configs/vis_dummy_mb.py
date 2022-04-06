# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial
from quant import parametric_d_xmax, gaussian_init, percentile_init


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = 'MobileNetV2_100'
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
  config.augment_name = 'plain'

  config.optimizer = 'rmsprop'
  config.learning_rate = 0.0000125  # 0.0001
  config.lr_boundaries_scale = None
  config.warmup_epochs = 2.0
  config.momentum = 0.9
  config.batch_size = 1024
  config.eval_batch_size = 4096
  config.weight_decay = 0.00001
  config.nesterov = True
  config.smoothing = .1

  config.num_epochs = 50
  config.log_every_steps = 256

  config.cache = True

  # Load pretrained weights.
  config.pretrained = None
  config.pretrained_quant = '..'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()

  config.quant_target.weight_mb = 1731.0
  config.quant_target.weight_penalty = .0001
  config.quant_target.act_mode = 'sum'
  config.quant_target.act_mb = 2524.0  # 2505.0
  config.quant_target.act_penalty = .0001
  config.quant_target.size_div = 8. * 1000.
  config.quant_target.eval_start = 61000  # 31050
  config.quant_target.update_every = 20  # every x steps d and xmax are updated

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = 4

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(
      parametric_d_xmax, init_fn=gaussian_init, bitwidth_min=1)

  # Conv in InvertedResidual blocks.
  config.quant.invertedresidual = ml_collections.ConfigDict()
  config.quant.invertedresidual.weight = partial(
      parametric_d_xmax, init_fn=gaussian_init, bitwidth_min=1)
  config.quant.invertedresidual.act = partial(parametric_d_xmax, act=True, init_fn=partial(
      percentile_init, perc=99.9), bitwidth_min=1, d_max=8)

  # Conv for head layer.
  config.quant.head = ml_collections.ConfigDict()
  config.quant.head.weight = partial(
      parametric_d_xmax, init_fn=gaussian_init, bitwidth_min=1)
  config.quant.head.act = partial(parametric_d_xmax, act=True, init_fn=partial(
      percentile_init, perc=99.9), bitwidth_min=1, d_max=8)

  # Average quant.
  config.quant.average = partial(parametric_d_xmax, act=True, init_fn=partial(
      percentile_init, perc=99.9), bitwidth_min=1, d_max=8)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(
      parametric_d_xmax, init_fn=gaussian_init, bitwidth_min=1)
  config.quant.dense.act = partial(parametric_d_xmax, act=True, init_fn=partial(
      percentile_init, perc=99.9), bitwidth_min=1, d_max=8)
  config.quant.dense.bias = partial(
      parametric_d_xmax, init_fn=gaussian_init, bitwidth_min=1)

  return config
