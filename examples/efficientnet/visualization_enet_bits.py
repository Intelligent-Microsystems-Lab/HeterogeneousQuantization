# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import functools
import time

from absl import app
from absl import flags
from absl import logging
from clu import platform

from flax import jax_utils


import matplotlib.pyplot as plt

from flax.training import common_utils

import jax
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds

import jax.numpy as jnp
import numpy as np

import ml_collections
from ml_collections import config_flags

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import models
from train_utils import (
    TrainState,
    create_model,
    create_train_state,
    restore_checkpoint,
    eval_step
)

# import resource
# low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)


def plot_bits(config: ml_collections.ConfigDict,workdir: str):
  rng = random.PRNGKey(config.seed)

  model_cls = getattr(models, config.model)
  model = create_model(model_cls=model_cls, num_classes=config.num_classes, config=config)

  rng, subkey = jax.random.split(rng, 2)
  state = create_train_state(
      subkey, config, model, config.image_size, lambda x: x)
  state = restore_checkpoint(state, workdir)


  xmax = jax.tree_util.tree_flatten(state.params['quant_params'])[0][::2]
  d = jax.tree_util.tree_flatten(state.params['quant_params'])[0][1::2]
  

  font_size = 22

  fig, ax = plt.subplots(nrows = 2, ncols =1, figsize=(12, 8.8))
  ax[0].spines["top"].set_visible(False)
  ax[0].spines["right"].set_visible(False)
  ax[1].spines["top"].set_visible(False)
  ax[1].spines["right"].set_visible(False)

  ax[0].xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax[0].yaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax[1].xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax[1].yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax[0].spines[axis].set_linewidth(5)
    ax[1].spines[axis].set_linewidth(5)

  # for tick in ax.xaxis.get_major_ticks():
  #   tick.label1.set_fontweight('bold')
  # for tick in ax.yaxis.get_major_ticks():
  #   tick.label1.set_fontweight('bold')

  import pdb; pdb.set_trace()
  ax[0].bar(np.arange(len(xmax)), xmax)
  ax[1].bar(np.arange(len(d)), d)

  ax[0].set_xlabel("Layer", fontsize=font_size, fontweight='bold')
  ax[0].set_ylabel("Bit Width", fontsize=font_size, fontweight='bold')
  # plt.legend(
  #     bbox_to_anchor=(0.5, 1.2),
  #     loc="upper center",
  #     ncol=2,
  #     frameon=False,
  #     prop={'weight': 'bold', 'size': font_size}
  # )
  plt.tight_layout()
  plt.savefig('efficientnet/figures/bitwidths.png')
  plt.close()



def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d',
               jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'proc_index: {jax.process_index()}, '
                                       f'proc_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')

  plot_bits(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
