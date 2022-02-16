# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Tests for MobileNetV2."""

from absl.testing import absltest

import jax
from jax import numpy as jnp
import numpy as np

from mobilenetv2.configs import mobilenetv2_fp32 as default_lib

import mobilenetv2.models as models

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_most_optimizations', True)


class MobileNetV2Test(absltest.TestCase):
  """Test cases for ResNet v1 model definition."""

  def test_mobilenetv2_model(self):
    """Tests MobileNetV2 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    config = default_lib.get_config()
    model_def = models.MobileNetV2_100(
        num_classes=1000, dtype=jnp.float32, config=config)
    rng, prng = jax.random.split(rng, 2)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.float32), rng=prng, train=False)

    # check total number of parameters
    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), 3504872)

    # variables params and batch stats
    self.assertLen(variables, 2)

    # MobileNetV2 model will create parameters for the following layers:
    #   stem conv + batch_norm = 2
    #   InvertedResidual: [1, 2, 3, 4, 3, 3, 1] = 17
    #   head conv + batch_norm = 2
    #   Followed by a Dense layer = 1
    self.assertLen(variables['params'], 22)


if __name__ == '__main__':
  absltest.main()
