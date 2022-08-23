# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Default Hyperparameter configuration."""

import ml_collections
from functools import partial
from quant import parametric_d_xmax, gaussian_init, percentile_init, round_ewgs, round_invtanh


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.seed = 203853699

  # As defined in the `models` module.
  config.model = 'MnistNetB0'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'mnist'
  config.num_classes = 10
  config.tfds_data_dir = None # 'gs://imagenet_clemens/tensorflow_datasets'
  config.image_size = 28
  config.crop_padding = 0

  # Mean and std style for pre-processing.
  # config.mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  # config.stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

  # Edge models use inception-style MEAN & STDDEV for better post-quantization.
  config.mean_rgb = [127.0]
  config.stddev_rgb = [128.0]
  config.augment_name = 'plain'

  config.optimizer = 'rmsprop'
  config.learning_rate = 0.001 # 0.0001
  config.lr_boundaries_scale = None
  config.warmup_epochs = 2.0
  config.momentum = 0.9
  config.batch_size = 512
  config.eval_batch_size = 4096
  config.weight_decay = 0.00001
  config.nesterov = True
  config.smoothing = 0.

  config.num_epochs = 20
  config.log_every_steps = 256

  config.cache = True

  # Load pretrained weights.
  config.pretrained = None  # "../../pretrained_efficientnet/enet-lite0_best"
  # "../../pretrained_efficientnet/efficientnet-lite0"
  config.pretrained_quant = None #"gs://imagenet_clemens/enet-lite0_pre/efficientnet-lite0_mixed_bits_5"

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  config.quant_target = ml_collections.ConfigDict()
  config.quant_target.size_div = 8. * 1000.
  config.quant_target.update_every = 1

  # config.quant_target.weight_mb = 1731.0
  # config.quant_target.weight_penalty = .0001
  # config.quant_target.act_mode = 'sum'
  # config.quant_target.act_mb = 2524.0  # 2505.0
  # config.quant_target.act_penalty = .0001
  # config.quant_target.size_div = 8. * 1000.
  # config.quant_target.eval_start = 61000  # 31050
  # config.quant_target.update_every = 20  # every x steps d and xmax are updated

  config.quant = ml_collections.ConfigDict()

  config.quant.bits = None

  config.quant.g_scale = 5e-3


  # Conv in MBConv blocks.
  config.quant.mbconv = ml_collections.ConfigDict()
  #config.quant.mbconv.weight = ml_collections.ConfigDict()
  #config.quant.mbconv.act = ml_collections.ConfigDict()

  # Final linear layer.
  config.quant.dense = ml_collections.ConfigDict()
  #config.quant.dense.weight = ml_collections.ConfigDict()
  #config.quant.dense.act = ml_collections.ConfigDict()
  #config.quant.dense.bias = ml_collections.ConfigDict()

  return config
