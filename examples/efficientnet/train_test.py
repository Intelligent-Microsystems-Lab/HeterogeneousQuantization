# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Tests for flax.examples.imagenet.train."""

import tempfile
import importlib

from absl.testing import absltest

import jax
from jax import random

import tensorflow as tf
import tensorflow_datasets as tfds

# Local imports.
import models
import train
import train_util

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_most_optimizations', True)


class TrainTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

  def test_create_model(self):
    """Tests creating model."""
    config_module = importlib.import_module('configs.efficientnet-lite0')
    config = config_module.get_config()
    model = train_util.create_model(
        model_cls=models.EfficientNetB0,  # pylint: disable=protected-access
        num_classes=1000,
        config=config)
    params, quant_params, batch_stats = train_util.initialized(
        random.PRNGKey(0), 224, model)
    variables = {'params': params,
                 'quant_params': quant_params, 'batch_stats': batch_stats}
    x = random.normal(random.PRNGKey(1), (8, 224, 224, 3))
    y = model.apply(variables, x, rng=jax.random.PRNGKey(0), train=False)
    self.assertEqual(y.shape, (8, 1000))

  def test_train_and_evaluate(self):
    """Tests training and evaluation loop using mocked data."""
    # Create a temporary directory where tensorboard metrics are written.
    workdir = tempfile.mkdtemp()
    data_dir = '../../unit_tests/tensorflow_datasets'

    # Define training configuration
    config_module = importlib.import_module('configs.efficientnet-lite0')
    config = config_module.get_config()
    config.model = 'EfficientNetB0'
    config.dataset = 'imagenet2012'
    config.batch_size = 16
    config.num_epochs = 1
    config.num_train_steps = 1
    config.steps_per_eval = 1

    with tfds.testing.mock_data(num_examples=1, data_dir=data_dir):
      train.train_and_evaluate(workdir=workdir, config=config)


if __name__ == '__main__':
  absltest.main()
