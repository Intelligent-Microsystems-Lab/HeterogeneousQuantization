# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import ml_collections
import datetime


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  config.batch_size = 16.
  config.num_epochs = 30.
  config.learning_rate = 0.0005

  # PC parameters
  config.infer_lr = 0.2
  config.infer_steps = 50.

  config.quant = {
      # noise
      "weight_noise": 0.0,
      "act_noise": 0.0,
      "weight_bwd_noise": 0.0,
      "act_bwd_noise": 0.0,
      "val_noise": 0.0,
      "err_inpt_noise": 0.0,
      "err_weight_noise": 0.0,
      # quant
      "weight_bits": 5.0,
      "act_bits": 5.0,
      "weight_bwd_bits": 5.0,
      "act_bwd_bits": 5.0,
      "val_bits": 5.0,
      "err_inpt_bits": 5.0,
      "err_weight_bits": 5.0,
  }

  # data set
  config.ds = "cifar10"
  config.num_classes = 10
  config.ds_xdim = 32
  config.ds_ydim = 32
  config.ds_channels = 3

  return config
