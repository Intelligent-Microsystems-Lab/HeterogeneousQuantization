# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Tests for EfficientNet."""

from absl.testing import absltest

import jax
from jax import numpy as jnp
import numpy as np

import models


jax.config.update('jax_disable_most_optimizations', True)
jax.config.update('jax_platform_name', 'cpu')


class EfficientNetTest(absltest.TestCase):
  """Test cases for ResNet v1 model definition."""

  def test_efficienteet_b0_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.EfficientNetB0(num_classes=1000, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.float32), train=False)

    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), 4652008)
    self.assertLen(variables, 2)

  def test_efficienteet_b1_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.EfficientNetB1(num_classes=1000, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 240, 240, 3), jnp.float32), train=False)

    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), 5416680)
    self.assertLen(variables, 2)

  def test_efficienteet_b2_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.EfficientNetB2(num_classes=1000, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 260, 260, 3), jnp.float32), train=False)

    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), 6092072)
    self.assertLen(variables, 2)

  def test_efficienteet_b3_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.EfficientNetB3(num_classes=1000, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 280, 280, 3), jnp.float32), train=False)

    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), 8197096)
    self.assertLen(variables, 2)

  def test_efficienteet_b4_model(self):
    """Tests ResNet V1 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    model_def = models.EfficientNetB4(num_classes=1000, dtype=jnp.float32)
    variables = model_def.init(
        rng, jnp.ones((8, 300, 300, 3), jnp.float32), train=False)

    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), 13006568)
    self.assertLen(variables, 2)


if __name__ == '__main__':
  absltest.main()
