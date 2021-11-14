# IMSL Lab - University of Notre Dame
# Copied from https://github.com/google/flax/

from absl.testing import absltest

from flax import linen as nn

import jax
from jax import random
from jax import test_util as jtu
from jax.nn import initializers
import jax.numpy as jnp

import numpy as np

from batchnorm import BatchNorm

class NormalizationTest(absltest.TestCase):

  def test_batch_norm(self):
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    x = random.normal(key1, (4, 3, 2))
    model_cls = BatchNorm(momentum=0.9, use_running_average=False)
    y, initial_params = model_cls.init_with_output(key2, x)

    mean = y.mean((0, 1))
    var = y.var((0, 1))
    np.testing.assert_allclose(mean, np.array([0., 0.]), atol=1e-4)
    np.testing.assert_allclose(var, np.array([1., 1.]), rtol=1e-4)

    y, vars_out = model_cls.apply(initial_params, x, mutable=['batch_stats'])

    ema = vars_out['batch_stats']
    np.testing.assert_allclose(
        ema['mean'], 0.1 * x.mean((0, 1), keepdims=False), atol=1e-4)
    np.testing.assert_allclose(
        ema['var'], 0.9 + 0.1 * x.var((0, 1), keepdims=False), rtol=1e-4)

  def test_batch_norm_multi_init(self):
    class Foo(nn.Module):
      @nn.compact
      def __call__(self, x):
        norm = BatchNorm(
            name="norm",
            use_running_average=False,
            axis_name="batch",
        )
        x = norm(x)
        return x, norm(x)

    key = random.PRNGKey(0)
    model = Foo()
    x = random.normal(random.PRNGKey(1), (2, 4))
    (y1, y2), variables = model.init_with_output(key, x)
    np.testing.assert_allclose(y1, y2, rtol=1e-4)

if __name__ == '__main__':
  absltest.main()