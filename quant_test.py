# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Unit Test for quant


from absl.testing import absltest
from absl.testing import parameterized
from flax.core import freeze, unfreeze


from jax import random
import jax
import jax.numpy as jnp


import numpy as np
import re


from quant import signed_uniform_max_scale_quant_ste, parametric_d

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_most_optimizations', True)


def signed_uniform_max_scale_quant_ste_equality_data():
  return (
      dict(
          x_dim=100,
          y_dim=30,
          dtype=jnp.int8,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          dtype=jnp.int16,
      ),
  )


def signed_uniform_max_scale_quant_ste_unique_data():
  return (
      dict(
          x_dim=100,
          y_dim=30,
          bits=2,
          scale=23,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=3,
          scale=3231,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=4,
          scale=39,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=5,
          scale=913,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=6,
          scale=4319,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=7,
          scale=0.124,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=8,
          scale=3,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=9,
          scale=780,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=10,
          scale=0.01324,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=11,
          scale=781,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=12,
          scale=4561,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=13,
          scale=813,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=14,
          scale=9013,
      ),
      dict(
          x_dim=100,
          y_dim=30,
          bits=15,
          scale=561,
      ),
  )


class QuantOpsTest(parameterized.TestCase):
  @parameterized.product(
      signed_uniform_max_scale_quant_ste_equality_data(),
      quantizer=(signed_uniform_max_scale_quant_ste, parametric_d)
  )
  def test_equality_native_dtypes(
      self, x_dim, y_dim, dtype, quantizer,
  ):
    key = random.PRNGKey(8627169)

    key, subkey = jax.random.split(key)
    data = jax.random.randint(
        subkey,
        (x_dim, y_dim),
        minval=np.iinfo(dtype).min,
        maxval=np.iinfo(dtype).max,
    )
    data = data.at[0, 0].set(np.iinfo(dtype).min)
    data = jnp.clip(
        data, a_min=np.iinfo(dtype).min + 1, a_max=np.iinfo(dtype).max
    )
    data = jnp.array(data, jnp.float64)

    bits = int(re.split("(\d+)", dtype.__name__)[1])  # noqa: W605

    key, subkey = jax.random.split(key)
    variables = quantizer(bits).init(subkey, data)

    scale = 1
    if 'quant_params' in variables:
      if 'step_size' in variables['quant_params']:
        scale = variables['quant_params']['step_size']

    dataq = quantizer(bits).apply(variables, data * scale)

    np.testing.assert_allclose(data, dataq / scale)

  @parameterized.product(
      signed_uniform_max_scale_quant_ste_unique_data(),
      quantizer=(signed_uniform_max_scale_quant_ste, parametric_d)
  )
  def test_unique_values(
      self, x_dim, y_dim, bits, scale, quantizer
  ):
    key = random.PRNGKey(8627169)

    key, subkey = jax.random.split(key)
    data = (
        jax.random.uniform(subkey, (1024, 1024), minval=-1, maxval=1)
        * scale
    )
    data = data.at[0, 0].set(scale)

    key, subkey = jax.random.split(key)
    variables = quantizer(bits).init(subkey, data)

    if 'quant_params' in variables:
      if 'step_size' in variables['quant_params']:
        variables = unfreeze(variables)
        variables['quant_params']['step_size'] = scale / (2 ** (bits - 1) - 1)
        variables = freeze(variables)

    dataq = quantizer(bits).apply(variables, data)

    self.assertEqual(
        len(np.unique(dataq)), ((2 ** (bits - 1) - 1) * 2) + 1
    )


if __name__ == "__main__":
  absltest.main()
