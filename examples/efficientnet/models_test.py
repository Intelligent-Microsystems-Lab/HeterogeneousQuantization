# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Tests for flax.examples.imagenet.models."""

from absl.testing import absltest

import jax
from jax import numpy as jnp

import models


jax.config.update('jax_disable_most_optimizations', True)


class ResNetV1Test(absltest.TestCase):
  """Test cases for ResNet v1 model definition."""

  def test_resnet_v1_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.ResNet50(num_classes=10, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.float32))

    self.assertLen(variables, 2)
    # Resnet50 model will create parameters for the following layers:
    #   conv + batch_norm = 2
    #   BottleneckResNetBlock in stages: [3, 4, 6, 3] = 16
    #   Followed by a Dense layer = 1
    self.assertLen(variables['params'], 19)


if __name__ == '__main__':
  absltest.main()
