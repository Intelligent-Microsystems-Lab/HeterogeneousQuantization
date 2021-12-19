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
  config.model = 'SNN'

  config.dataset = 'DVSGesture'
  config.data_dir = '../../../data_dvs_gesture/dvs_gestures_build19.hdf5'
  config.chunk_size_train = 1800
  config.chunk_size_test = 500
  config.dt = 1000

  config.num_classes = 11

  config.optimizer = 'sgd'
  config.learning_rate = 0.01  # 0.0001
  config.lr_boundaries_scale = None
  config.warmup_epochs = 5  # for optimizer to settle in
  config.weight_decay = 1e-5
  config.momentum = 0.9
  config.batch_size = 72
  config.eval_batch_size = 128
  config.smoothing = .1

  config.num_epochs = 50
  config.log_every_steps = 5
  config.num_devices = None

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()
  config.quant_target.size_div = 8. * 1024.

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = 32

  config.quant.g_scale = 0.

  # Conv for stem layer.
  config.quant.stem = ml_collections.ConfigDict()

  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()

  # Conv for head layer.
  config.quant.head = ml_collections.ConfigDict()

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()

  return config