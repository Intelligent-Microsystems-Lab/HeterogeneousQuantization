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
  config.model = 'EfficientNetB2'

  # `name` argument of tensorflow_datasets.builder()
  config.cache = True
  config.dataset = 'imagenet2012'
  config.tfds_data_dir = None  # 'gs://imagenet_clemens/tensorflow_datasets'
  config.image_size = 260
  config.crop_padding = 32

  # Mean and std style for pre-processing.
  # config.mean_rgb = [0.485 * 255, 0.456 * 255, 0.406 * 255]
  # config.stddev_rgb = [0.229 * 255, 0.224 * 255, 0.225 * 255]

  # Edge models use inception-style MEAN & STDDEV for better post-quantization.
  config.mean_rgb = [127.0, 127.0, 127.0]
  config.stddev_rgb = [128.0, 128.0, 128.0]

  config.num_classes = 1000

  # Load pretrained weights.
  config.pretrained = "../../../pretrained_efficientnet/efficientnet-lite2"

  config.learning_rate = 0.016
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 2048

  config.num_epochs = 350
  config.log_every_steps = 256

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
