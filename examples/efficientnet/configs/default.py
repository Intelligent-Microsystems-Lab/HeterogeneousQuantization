# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = 'ResNet18'

  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenette'
  config.image_size = 224
  config.num_classes = 10

  config.learning_rate = 0.1
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 128

  config.num_epochs = 100.0
  config.log_every_steps = 100

  config.cache = False

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant = ml_collections.ConfigDict()

  # noise
  config.quant.weight_noise = 0.0
  config.quant.act_noise = 0.0
  config.quant.weight_bwd_noise = 0.0
  config.quant.act_bwd_noise = 0.0
  config.quant.val_noise = 0.0
  config.quant.err_inpt_noise = 0.0
  config.quant.err_weight_noise = 0.0
  
  # quant
  config.quant.weight_bits = 5.0
  config.quant.act_bits = 5.0
  config.quant.weight_bwd_bits = 5.0
  config.quant.act_bwd_bits = 5.0
  config.quant.val_bits = 5.0
  config.quant.err_inpt_bits = 5.0
  config.quant.err_weight_bits = 5.0

  return config
