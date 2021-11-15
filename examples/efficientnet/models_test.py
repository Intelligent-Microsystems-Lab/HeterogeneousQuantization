# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""Tests for EfficientNet."""

from absl.testing import absltest
from absl.testing import parameterized

import importlib

import tensorflow as tf
import jax
from jax import numpy as jnp
import numpy as np
import optax

import models
from input_pipeline import preprocess_for_eval
from load_pretrained_weights import load_pretrained_weights
from train_util import TrainState


jax.config.update('jax_disable_most_optimizations', True)
jax.config.update('jax_platform_name', 'cpu')


def net_size_data():
  return (
      dict(
          testcase_name="EfficientNetB0",
          name="EfficientNetB0",
          param_count=4652008,
          inpt_size=224,
          rtol=1e-7,
          atol=1e-4,
      ),
      dict(
          testcase_name="EfficientNetB1",
          name="EfficientNetB1",
          param_count=5416680,
          inpt_size=240,
          rtol=1e-7,
          atol=1e-4,
      ),
      dict(
          testcase_name="EfficientNetB2",
          name="EfficientNetB2",
          param_count=6092072,
          inpt_size=260,
          rtol=1e-7,
          atol=1e-4,
      ),
      dict(
          testcase_name="EfficientNetB3",
          name="EfficientNetB3",
          param_count=8197096,
          inpt_size=280,
          rtol=1e-7,
          atol=1e-4,
      ),
      dict(
          testcase_name="EfficientNetB4",
          name="EfficientNetB4",
          param_count=13006568,
          inpt_size=300,
          rtol=1e-7,
          atol=1e-4,
      ),
  )


class EfficientNetTest(parameterized.TestCase):
  """Test cases for ResNet v1 model definition."""

  @parameterized.named_parameters(*net_size_data())
  def test_efficienteet_model(self, name, param_count, inpt_size, rtol, atol):
    """Tests EfficientNet model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    rng, prng = jax.random.split(rng, 2)
    model_cls = getattr(models, name)

    # get correct config
    config_module = importlib.import_module(
        'configs.efficientnet-lite' + str(name[-1]))
    config = config_module.get_config()

    model_def = model_cls(num_classes=1000, dtype=jnp.float32, config=config)
    variables = model_def.init(
        rng, jnp.ones((8, inpt_size, inpt_size, 3), jnp.float32),
        rng=prng, train=False)

    self.assertEqual(np.sum(jax.tree_util.tree_leaves(jax.tree_map(
        lambda x: np.prod(x.shape), variables['params']))), param_count)
    self.assertLen(variables, 2)

  @parameterized.named_parameters(*net_size_data())
  def test_efficienteet_inference(self, name, param_count, inpt_size,
                                  rtol, atol):

    # get correct config
    config_module = importlib.import_module(
        'configs.efficientnet-lite' + str(name[-1]))
    config = config_module.get_config()

    # initialize network
    rng = jax.random.PRNGKey(0)
    rng, prng1, prng2 = jax.random.split(rng, 3)
    model_cls = getattr(models, name)
    model_def = model_cls(num_classes=1000, dtype=jnp.float32, config=config)
    variables = model_def.init(
        prng1, jnp.ones((8, inpt_size, inpt_size, 3), jnp.float32),
        rng=prng2, train=False)

    # load pretrain weights
    tx = optax.rmsprop(0.0)
    if 'quant_params' in variables:
      quant_params = variables['quant_params']
    else:
      quant_params = {}
    state = TrainState.create(
        apply_fn=model_def.apply, params={
            'params': variables['params'],
            'quant_params': quant_params}, tx=tx,
        batch_stats=variables['batch_stats'],
        weight_size={}, act_size={})
    state = load_pretrained_weights(state, config.pretrained)

    # load inpt
    inpt_bytes = tf.io.read_file('../../../unit_tests/efficientnet/panda.jpg')
    inpt = np.reshape(preprocess_for_eval(
        inpt_bytes, config), (1, inpt_size, inpt_size, 3))

    # run inference
    rng, prng = jax.random.split(rng, 2)
    _, state = model_def.apply({'params': state.params['params'],
                                'quant_params': state.params['quant_params'],
                                'batch_stats': state.batch_stats,
                                'weight_size': state.weight_size,
                                'act_size': state.act_size}, inpt,
                               mutable=['intermediates', 'batch_stats',
                                        'weight_size', 'act_size'],
                               rng=prng,
                               train=False)

    # testing for equality
    np.testing.assert_allclose(inpt, np.load(
        '../../../unit_tests/efficientnet/enet' + str(name[-1]) + '_inputs\
        .npy'))

    np.testing.assert_allclose(state['intermediates']['stem'][0], np.load(
        '../../../unit_tests/efficientnet/enet' + str(name[-1]) + '_stem.npy'),
        rtol=rtol, atol=atol)

    np.testing.assert_allclose(state['intermediates']['features0'][0], np.load(
        '../../../unit_tests/efficientnet/enet' + str(name[-1]) + '_features0\
        .npy'), rtol=rtol, atol=atol)

    np.testing.assert_allclose(state['intermediates']['head'][0], np.load(
        '../../../unit_tests/efficientnet/enet' + str(name[-1]) + '_head.npy'),
        rtol=rtol, atol=atol)


if __name__ == '__main__':
  absltest.main()
