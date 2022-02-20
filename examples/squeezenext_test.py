# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Tests for SqueezeNext."""

# from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from squeezenext.configs import sqnxt23_w2_fp32 as default_lib

import squeezenext.models as models

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_most_optimizations', True)


def net_size_data():
  return (
      dict(
          testcase_name="sqnxt23_w1",
          name="sqnxt23_w1",
          param_count=724056,
      ),
      dict(
          testcase_name="sqnxt23_w3d2",
          name="sqnxt23_w3d2",
          param_count=1511824,
      ),
      dict(
          testcase_name="sqnxt23_w2",
          name="sqnxt23_w2",
          param_count=2583752,
      ),
      dict(
          testcase_name="sqnxt23v5_w1",
          name="sqnxt23v5_w1",
          param_count=921816,
      ),
      dict(
          testcase_name="sqnxt23v5_w3d2",
          name="sqnxt23v5_w3d2",
          param_count=1953616,
      ),
      dict(
          testcase_name="sqnxt23v5_w2",
          name="sqnxt23v5_w2",
          param_count=3366344,
      ),
  )


class SqueezeNextTest(parameterized.TestCase):
  """Test cases for SqueeezNet model definition."""

  @parameterized.named_parameters(*net_size_data())
  def test_squeezenext_model(self, name, param_count):
    """Tests SqueezeNext model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    config = default_lib.get_config()
    model_cls = getattr(models, name)
    model_def = model_cls(
        num_classes=1000, dtype=jnp.bfloat16, config=config)
    rng, prng = jax.random.split(rng, 2)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.bfloat16), rng=prng, train=False)

    # check total number of parameters
    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), param_count)

    # variables params and batch stats
    self.assertLen(variables, 2)


if __name__ == '__main__':
  absltest.main()
