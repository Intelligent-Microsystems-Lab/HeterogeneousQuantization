# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # As defined in the `models` module.
  config.model = 'ResNet18'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012'
  config.tfds_data_dir = 'gs://imagenet_clemens/tensorflow_datasets'

  config.learning_rate = 0.1
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 1024

  config.num_epochs = 100.0
  config.log_every_steps = 100

  config.cache = True
  config.half_precision = False

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1
  return config
