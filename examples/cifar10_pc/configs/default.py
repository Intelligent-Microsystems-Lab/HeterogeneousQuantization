# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import ml_collections
import datetime


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # config.seed = 203853699
  # config.work_dir = (
  #     "../../../training_dir/cifar10_pc-{date:%Y-%m-%d_%H-%M-%S}/".format(
  #         date=datetime.datetime.now()
  #     )
  # )
  config.batch_size = 64
  config.num_epochs = 70
  # config.warmup_epochs = 5
  # config.momentum = 0.0
  config.learning_rate = 0.0005
  config.infer_lr = 0.2
  config.infer_steps = 100
  config.num_classes = 10
  # config.label_smoothing = 0.1

  # noise config
  config.weight_noise = 0.0
  config.act_noise = 0.0

  config.weight_bwd_noise = 0.0
  config.act_bwd_noise = 0.0

  config.val_noise = 0.0

  config.err_inpt_noise = 0.0
  config.err_weight_noise = 0.0

  return config
