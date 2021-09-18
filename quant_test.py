# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Unit Test for quant


from absl.testing import absltest
from absl.testing import parameterized


from jax import random
import jax
import jax.numpy as jnp


import numpy as np
import re


from quant import signed_uniform_max_scale_quant_ste


def signed_uniform_max_scale_quant_ste_equality_data():
  return (
      dict(
          testcase_name="int8",
          x_dim=100,
          y_dim=30,
          dtype=jnp.int8,
      ),
      dict(
          testcase_name="int16",
          x_dim=100,
          y_dim=30,
          dtype=jnp.int16,
      ),
  )


def signed_uniform_max_scale_quant_ste_unique_data():
  return (
      dict(
          testcase_name="2_bits",
          x_dim=100,
          y_dim=30,
          bits=2,
          scale=23,
      ),
      dict(
          testcase_name="3_bits",
          x_dim=100,
          y_dim=30,
          bits=3,
          scale=3231,
      ),
      dict(
          testcase_name="4_bits",
          x_dim=100,
          y_dim=30,
          bits=4,
          scale=39,
      ),
      dict(
          testcase_name="5_bits",
          x_dim=100,
          y_dim=30,
          bits=5,
          scale=913,
      ),
      dict(
          testcase_name="6_bits",
          x_dim=100,
          y_dim=30,
          bits=6,
          scale=4319,
      ),
      dict(
          testcase_name="7_bits",
          x_dim=100,
          y_dim=30,
          bits=7,
          scale=0.124,
      ),
      dict(
          testcase_name="8_bits",
          x_dim=100,
          y_dim=30,
          bits=8,
          scale=3,
      ),
      dict(
          testcase_name="9_bits",
          x_dim=100,
          y_dim=30,
          bits=9,
          scale=780,
      ),
      dict(
          testcase_name="10_bits",
          x_dim=100,
          y_dim=30,
          bits=10,
          scale=0.01324,
      ),
      dict(
          testcase_name="11_bits",
          x_dim=100,
          y_dim=30,
          bits=11,
          scale=781,
      ),
      dict(
          testcase_name="12_bits",
          x_dim=100,
          y_dim=30,
          bits=12,
          scale=4561,
      ),
      dict(
          testcase_name="13_bits",
          x_dim=100,
          y_dim=30,
          bits=13,
          scale=813,
      ),
      dict(
          testcase_name="14_bits",
          x_dim=100,
          y_dim=30,
          bits=14,
          scale=9013,
      ),
      dict(
          testcase_name="15_bits",
          x_dim=100,
          y_dim=30,
          bits=15,
          scale=561,
      ),
  )


class QuantOpsTest(parameterized.TestCase):
  @parameterized.named_parameters(
      *signed_uniform_max_scale_quant_ste_equality_data()
  )
  def test_signed_uniform_max_scale_quant_ste_equality(
      self, x_dim, y_dim, dtype
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

    dataq = signed_uniform_max_scale_quant_ste(
        data, int(re.split("(\d+)", dtype.__name__)[1])  # noqa: W605
    )

    np.testing.assert_array_equal(data, dataq)

  @parameterized.named_parameters(
      *signed_uniform_max_scale_quant_ste_unique_data()
  )
  def test_signed_uniform_max_scale_quant_ste_unique(
      self, x_dim, y_dim, bits, scale
  ):
    key = random.PRNGKey(8627169)

    key, subkey = jax.random.split(key)
    data = (
        jax.random.uniform(subkey, (1024, 1024), minval=-1, maxval=1)
        * scale
    )
    data = data.at[0, 0].set(scale)

    dataq = signed_uniform_max_scale_quant_ste(data, bits)

    self.assertEqual(
        len(np.unique(dataq)), ((2 ** (bits - 1) - 1) * 2) + 1
    )


if __name__ == "__main__":
  absltest.main()
