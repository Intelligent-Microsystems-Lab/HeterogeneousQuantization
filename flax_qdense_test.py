# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Unit Test for QuantDense


from absl.testing import absltest
from absl.testing import parameterized

from flax import linen as nn
from flax import optim
from flax.core import freeze
from jax import random
from jax.nn import initializers
import jax
import jax.numpy as jnp
from flax import jax_utils
import functools

import numpy as np

import ml_collections

from flax_qdense import QuantDense


class DQG(nn.Module):
  """A simple fully connected model with QuantDense"""

  @nn.compact
  def __call__(self, x, channels, config, rng):
    """Description of CNN forward pass
    Args:
        x: an array (inputs)
        channels: an array containing number of channels for each layer
        config: bit width for gradient in the backward pass
    Returns:
        An array containing the result.
    """
    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=channels[0],
        kernel_init=initializers.lecun_normal(),
        config=config,
    )(x, subkey)
    rng, subkey = jax.random.split(rng, 2)
    x = QuantDense(
        features=channels[1],
        kernel_init=initializers.lecun_normal(),
        config=config,
    )(x, subkey)
    return x


class Dlinen(nn.Module):
  """Same model as above but with nn.Dense"""

  @nn.compact
  def __call__(self, x, channels):
    """Description of CNN forward pass
    Args:
        x: an array (inputs)
        channels: an array containing number of channels for each layer
    Returns:
        An array containing the result.
    """
    x = nn.Dense(
        features=channels[0], kernel_init=initializers.lecun_normal()
    )(x)
    x = nn.Dense(
        features=channels[1], kernel_init=initializers.lecun_normal()
    )(x)
    return x


def create_optimizer(params, learning_rate):
  optimizer_def = optim.GradientDescent(learning_rate=learning_rate)
  optimizer = optimizer_def.create(params)
  return optimizer


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(labels * logits, axis=-1))


# Train step for QuantDense layer
def train_step_dense_quant_grad(optimizer, batch, out_channels, config, rng):
  """Train for a single step."""

  def loss_fn(params):
    logits = DQG().apply(
        {"params": params}, batch["image"], out_channels, config, rng
    )
    loss = cross_entropy_loss(logits, batch["label"])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer


# Train step for nn.Dense layer
def train_step_dense(optimizer, batch, out_channels):
  """Train for a single step."""

  def loss_fn(params):
    logits = Dlinen().apply(
        {"params": params}, batch["image"], out_channels
    )
    loss = cross_entropy_loss(logits, batch["label"])
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer


# Test data for QuantDense
def dense_test_data():
  return (
      dict(
          testcase_name="base_case",
          examples=512,
          inp_channels=100,
          channels=[20, 10],
          config=ml_collections.FrozenConfigDict({}),
          numerical_tolerance=0,
      ),
      dict(
          testcase_name="base_case_1024examples_1channel",
          examples=1024,
          inp_channels=1,
          channels=[1, 1],
          config=ml_collections.FrozenConfigDict({}),
          numerical_tolerance=0,
      ),
      dict(
          testcase_name="base_case_200channels",
          examples=256,
          inp_channels=1,
          channels=[200, 1],
          config=ml_collections.FrozenConfigDict({}),
          numerical_tolerance=1e-7,
      ),
      dict(
          testcase_name="base_case_zero_noise",
          examples=512,
          inp_channels=100,
          channels=[20, 10],
          config=ml_collections.FrozenConfigDict(
              {
                  "weight_noise": 0.0,
                  "act_noise": 0.0,
                  "err_inpt_noise": 0.0,
                  "err_weight_noise": 0.0,
              }
          ),
          numerical_tolerance=0,
      ),
  )


def dense_act_noise_data():
  return (
      dict(
          testcase_name="noise_act_01",
          examples=100000,
          inp_channels=100,
          out_channels=200,
          noise=0.01,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_act_05",
          examples=100000,
          inp_channels=300,
          out_channels=100,
          noise=0.05,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_act_10",
          examples=100000,
          inp_channels=200,
          out_channels=100,
          noise=0.1,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_act_20",
          examples=100000,
          inp_channels=200,
          out_channels=200,
          noise=0.2,
          numerical_tolerance=0.01,
      ),
  )


def dense_weight_noise_data():
  return (
      dict(
          testcase_name="noise_weight_01",
          examples=1000,
          inp_channels=1000,
          out_channels=2000,
          noise=0.01,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_weight_05",
          examples=1000,
          inp_channels=3000,
          out_channels=1000,
          noise=0.05,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_weight_10",
          examples=1000,
          inp_channels=2000,
          out_channels=1000,
          noise=0.1,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_weight_20",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=0.01,
      ),
  )


def dense_err_inpt_noise_data():
  return (
      dict(
          testcase_name="noise_err_inpt_01",
          examples=1000,
          inp_channels=1000,
          out_channels=2000,
          noise=0.01,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_err_inpt_05",
          examples=1000,
          inp_channels=3000,
          out_channels=1000,
          noise=0.05,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_err_inpt_10",
          examples=1000,
          inp_channels=2000,
          out_channels=1000,
          noise=0.1,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_err_inpt_20",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=0.01,
      ),
  )


def dense_err_weight_noise_data():
  return (
      dict(
          testcase_name="noise_weight_noise_01",
          examples=1000,
          inp_channels=1000,
          out_channels=2000,
          noise=0.01,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_weight_noise_05",
          examples=1000,
          inp_channels=3000,
          out_channels=1000,
          noise=0.05,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_weight_noise_10",
          examples=1000,
          inp_channels=2000,
          out_channels=1000,
          noise=0.1,
          numerical_tolerance=0.01,
      ),
      dict(
          testcase_name="noise_weight_noise_20",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=0.01,
      ),
  )


class QuantDenseTest(parameterized.TestCase):
  @parameterized.named_parameters(*dense_test_data())
  def test_QuantDense_vs_nnDense(
      self, examples, inp_channels, channels, config, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """

    # create initial data
    key = random.PRNGKey(8627169)
    key, subkey1, subkey2 = random.split(key, 3)
    data_x = random.uniform(
        subkey1,
        (jax.device_count(), examples, inp_channels),
        minval=-1,
        maxval=1,
    )
    data_y = random.uniform(
        subkey2,
        (jax.device_count(), examples, channels[1]),
        minval=-1,
        maxval=1,
    )

    # setup QuantDense
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    params_quant_grad = DQG().init(
        subkey1, jnp.take(data_x, 0, axis=0), channels, config, subkey2
    )["params"]
    optimizer_quant_grad = create_optimizer(params_quant_grad, 1)

    # setup nn.Dense with parameters from QuantDense
    params = freeze(
        {
            "Dense_0": params_quant_grad["QuantDense_0"],
            "Dense_1": params_quant_grad["QuantDense_1"],
        }
    )
    optimizer = create_optimizer(params, 1)

    # check that weights are initially equal for both layers
    assert (
        params_quant_grad["QuantDense_0"]["kernel"]
        == params["Dense_0"]["kernel"]
    ).all() and (
        params_quant_grad["QuantDense_1"]["kernel"]
        == params["Dense_1"]["kernel"]
    ).all(), "Initial parameters not equal"

    optimizer_quant_grad = jax_utils.replicate(optimizer_quant_grad)
    optimizer = jax_utils.replicate(optimizer)

    p_train_step_dense_quant_grad = jax.pmap(
        functools.partial(
            train_step_dense_quant_grad,
            out_channels=channels,
            config=config,
            rng=subkey3,
        ),
        axis_name="batch",
    )
    p_train_step_dense = jax.pmap(
        functools.partial(
            train_step_dense,
            out_channels=channels,
        ),
        axis_name="batch",
    )

    # one backward pass
    optimizer_quant_grad = p_train_step_dense_quant_grad(
        optimizer_quant_grad,
        {"image": data_x, "label": data_y},
    )
    optimizer = p_train_step_dense(
        optimizer,
        {"image": data_x, "label": data_y},
    )

    # determine difference between nn.Dense and QuantDense
    self.assertLessEqual(
        jnp.mean(
            (
                jnp.mean(
                    abs(
                        optimizer.target["Dense_1"]["kernel"]
                        - optimizer_quant_grad.target["QuantDense_1"][
                            "kernel"
                        ]
                    )
                )
                / jnp.mean(abs(optimizer.target["Dense_1"]["kernel"]))
            )
            + (
                jnp.mean(
                    abs(
                        optimizer.target["Dense_0"]["kernel"]
                        - optimizer_quant_grad.target["QuantDense_0"][
                            "kernel"
                        ]
                    )
                )
                / jnp.mean(abs(optimizer.target["Dense_0"]["kernel"]))
            )
        ),
        numerical_tolerance,
    )

  @parameterized.named_parameters(*dense_act_noise_data())
  def test_act_noise(
      self, examples, inp_channels, out_channels, noise, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """
    config = ml_collections.FrozenConfigDict(
        {
            "weight_noise": 0.0,
            "act_noise": noise,
            "err_inpt_noise": 0.0,
            "err_weight_noise": 0.0,
        }
    )

    key = random.PRNGKey(34835972)
    rng1, rng2, rng3 = jax.random.split(key, 3)

    data = jnp.ones((examples, inp_channels))

    test_dense = QuantDense(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)
    out_d = test_dense.apply(variables, data, rng3)

    # test for mean
    self.assertLessEqual(
        jnp.mean(out_d) - inp_channels, numerical_tolerance
    )

    # test for variance
    self.assertLessEqual(
        jnp.std(out_d)
        - jnp.sqrt((1 / 12 * (noise * 2) ** 2) * inp_channels),
        numerical_tolerance,
    )

  @parameterized.named_parameters(*dense_weight_noise_data())
  def test_weight_noise(
      self, examples, inp_channels, out_channels, noise, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """
    config = ml_collections.FrozenConfigDict(
        {
            "weight_noise": noise,
            "act_noise": 0.0,
            "err_inpt_noise": 0.0,
            "err_weight_noise": 0.0,
        }
    )

    key = random.PRNGKey(34835972)
    rng1, rng2, rng3 = jax.random.split(key, 3)

    data = jnp.ones((examples, inp_channels))

    test_dense = QuantDense(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)
    out_d = test_dense.apply(variables, data, rng3)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(out_d),
        inp_channels,
        rtol=1e-04,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(out_d),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * inp_channels),
        rtol=1e-01,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

  @parameterized.named_parameters(*dense_err_inpt_noise_data())
  def test_err_inpt_noise(
      self, examples, inp_channels, out_channels, noise, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """
    config = ml_collections.FrozenConfigDict(
        {
            "weight_noise": 0.0,
            "act_noise": 0.0,
            "err_inpt_noise": noise,
            "err_weight_noise": 0.0,
        }
    )

    key = random.PRNGKey(34835972)
    rng1, rng2, rng3 = jax.random.split(key, 3)

    data = jnp.ones((examples, inp_channels))

    test_dense = QuantDense(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)

    def loss_fn(data):
      out_d = test_dense.apply(variables, data, rng3)
      return jnp.sum(out_d)

    grads_wrt_inpt = jax.grad(loss_fn)(data)
    # grads flowing up are clean because the loss function is not squared.

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_inpt),
        out_channels,
        rtol=numerical_tolerance,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    np.testing.assert_allclose(
        jnp.std(grads_wrt_inpt),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * out_channels),
        rtol=1e-01,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

  @parameterized.named_parameters(*dense_err_weight_noise_data())
  def test_err_weight_noise(
      self, examples, inp_channels, out_channels, noise, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """
    config = ml_collections.FrozenConfigDict(
        {
            "weight_noise": 0.0,
            "act_noise": 0.0,
            "err_inpt_noise": 0.0,
            "err_weight_noise": noise,
        }
    )

    key = random.PRNGKey(34835972)
    rng1, rng2, rng3 = jax.random.split(key, 3)

    data = jnp.ones((examples, inp_channels))

    test_dense = QuantDense(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)

    def loss_fn(params):
      out_d = test_dense.apply(params, data, rng3)
      return jnp.sum(out_d)

    grads_wrt_inpt = jax.grad(loss_fn)(variables)
    # grads flowing up are clean because the loss function is not squared.

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_inpt["params"]["kernel"]),
        examples,
        rtol=1e-04,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(grads_wrt_inpt["params"]["kernel"]),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * examples),
        rtol=1e-01,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

  @parameterized.named_parameters(*dense_act_noise_data())
  def test_act_bwd_noise(
      self, examples, inp_channels, out_channels, noise, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """
    config = ml_collections.FrozenConfigDict(
        {
            "weight_noise": 0.0,
            "act_bwd_noise": noise,
            "err_inpt_noise": 0.0,
            "err_weight_noise": 0.0,
        }
    )

    key = random.PRNGKey(34835972)
    rng1, rng2, rng3 = jax.random.split(key, 3)

    data = jnp.ones((examples, inp_channels))

    test_dense = QuantDense(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)

    # test for
    def loss_fn(params):
      out_d = test_dense.apply(params, data, rng3)
      return jnp.sum(out_d)

    grads_wrt_weights = jax.grad(loss_fn)(variables)
    # grads flowing up are clean because the loss function is not squared.

    # test for mean
    self.assertLessEqual(
        jnp.mean(grads_wrt_weights["params"]["kernel"]) - examples,
        numerical_tolerance * examples / inp_channels,
    )

    # test for variance
    self.assertLessEqual(
        jnp.std(grads_wrt_weights["params"]["kernel"])
        - jnp.sqrt((1 / 12 * (noise * 2) ** 2) * examples),
        numerical_tolerance * examples / inp_channels,
    )

  @parameterized.named_parameters(*dense_weight_noise_data())
  def test_weight_bwd_noise(
      self, examples, inp_channels, out_channels, noise, numerical_tolerance
  ):
    """
    Unit test to check whether QuantDense does exactly the same as
    nn.Dense when gradient quantization is turned off.
    """
    config = ml_collections.FrozenConfigDict(
        {
            "weight_bwd_noise": noise,
            "act_noise": 0.0,
            "err_inpt_noise": 0.0,
            "err_weight_noise": 0.0,
        }
    )

    key = random.PRNGKey(34835972)
    rng1, rng2, rng3 = jax.random.split(key, 3)

    data = jnp.ones((examples, inp_channels))

    test_dense = QuantDense(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)

    # test for
    def loss_fn(data):
      out_d = test_dense.apply(variables, data, rng3)
      return jnp.sum(out_d)

    grads_wrt_inpt = jax.grad(loss_fn)(data)
    # grads flowing up are clean because the loss function is not squared.

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_inpt),
        out_channels,
        rtol=1e-04,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(grads_wrt_inpt),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * out_channels),
        rtol=1e-01,
        atol=0,
        equal_nan=True,
        err_msg="",
        verbose=True,
    )


if __name__ == "__main__":
  absltest.main()
