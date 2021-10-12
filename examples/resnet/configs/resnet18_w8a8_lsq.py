# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial

from quant import signed_uniform_max_scale_quant_ste, parametric_d


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 1349194

  # As defined in the `models` module.
  config.model = 'ResNet18'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012'
  config.tfds_data_dir = 'gs://imagenet_clemens/tensorflow_datasets'

  config.learning_rate = 0.001
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 1024
  config.weight_decay = 0.0001

  config.num_epochs = 100.0
  config.log_every_steps = 100

  config.cache = True
  config.half_precision = False

  config.pretrained = '../../../pretrained_resnet/resnet18'

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant = ml_collections.ConfigDict()

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()
  config.quant.stem.weight = partial(parametric_d, bits=8)
  config.quant.stem.act = partial(parametric_d, bits=8)

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  config.quant.mbconv.weight = partial(parametric_d, bits=8)
  config.quant.mbconv.act = partial(parametric_d, bits=8)

  # Average quant.
  config.quant.average = partial(parametric_d, bits=8)

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  config.quant.dense.weight = partial(parametric_d, bits=8)
  config.quant.dense.act = partial(parametric_d, bits=8)
  config.quant.dense.bias = partial(parametric_d, bits=8)

  return config
