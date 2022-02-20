# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Tests for SqueezeNext."""

# from absl import logging
from absl.testing import absltest
from absl.testing import parameterized

# import importlib

import jax
from jax import numpy as jnp
import numpy as np
# import optax

# from PIL import Image
# from torchvision import transforms

from squeezenext.configs import sqnxt23_w2_fp32 as default_lib

import squeezenext.models as models
# from squeezenext.squeezenext_load_pretrained_weights import (
#     squeezenext_load_pretrained_weights
# )

# from train_utils import TrainState  # noqa: E402

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
  """Test cases for SqueeezNet v1 model definition."""

  @parameterized.named_parameters(*net_size_data())
  def test_squeezenext_model(self, name, param_count):
    """Tests SqueezeNext v1.1 model definition and output (variables)."""
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

  # def test_squeezenext_inference(self):

  #   # get correct config
  #   config_module = importlib.import_module(
  #       'configs.squeezenext_v11_fp32')
  #   config = config_module.get_config()

  #   # initialize network
  #   rng = jax.random.PRNGKey(0)
  #   rng, prng1, prng2 = jax.random.split(rng, 3)
  #   model_cls = getattr(models, config.model)
  #   model_def = model_cls(num_classes=1000, dtype=jnp.float32, config=config)
  #   variables = model_def.init(
  #       prng1, jnp.ones((8, 224, 224, 3), jnp.float32),
  #       rng=prng2, train=False)

  #   # load pretrain weights
  #   tx = optax.rmsprop(0.0)
  #   if 'quant_params' in variables:
  #     quant_params = variables['quant_params']
  #   else:
  #     quant_params = {}
  #   state = TrainState.create(
  #       apply_fn=model_def.apply, params={
  #           'params': variables['params'],
  #           'quant_params': quant_params}, tx=tx,
  #       batch_stats=variables['batch_stats'],
  #       weight_size={}, act_size={}, quant_config={})
  #   state = squeezenext_load_pretrained_weights(state, config.pretrained)

  #   # load inpt
  #  input_image = Image.open('../../unit_tests/squeezenext_unit_test/dog.jpg')
  #   preprocess = transforms.Compose([
  #       transforms.Resize(256),
  #       transforms.CenterCrop(224),
  #       transforms.ToTensor(),
  #       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
  #                            0.229, 0.224, 0.225]),
  #   ])
  #   input_tensor = preprocess(input_image)
  #   input_batch = input_tensor.unsqueeze(0)
  #   input_batch = jnp.moveaxis(
  #       jnp.array(input_batch), (0, 1, 2, 3), (0, 3, 1, 2))

  #   # run inference
  #   rng, prng = jax.random.split(rng, 2)
  #   _, state = model_def.apply({'params': state.params['params'],
  #                               'quant_params': state.params['quant_params'],
  #                               'batch_stats': state.batch_stats,
  #                               'weight_size': state.weight_size,
  #                               'act_size': state.act_size}, input_batch,
  #                              mutable=['intermediates', 'batch_stats',
  #                                       'weight_size', 'act_size'],
  #                              rng=prng,
  #                              train=False)

  #   rtol = 1e-3
  #   atol = 1e-4
  #   # testing for equality
  #   logging.info('Check stem.')
  #   pytorch_stem = np.load('../../unit_tests/squeezenext_unit_test/stem.npy')
  #   pytorch_stem = jnp.moveaxis(
  #       jnp.array(pytorch_stem), (0, 1, 2, 3), (0, 3, 1, 2))
  #   np.testing.assert_allclose(
  #      state['intermediates']['stem'][0], pytorch_stem, rtol=rtol, atol=atol)

  #   for i in range(8):
  #     logging.info('Check feature: ' + str(i))
  #     pytorch_features = np.load(
  #       '../../unit_tests/squeezenext_unit_test/features_' + str(i) + '.npy')
  #     pytorch_features = jnp.moveaxis(
  #         jnp.array(pytorch_features), (0, 1, 2, 3), (0, 3, 1, 2))
  #     np.testing.assert_allclose(
  #         state['intermediates']['fire_' + str(i)][0], pytorch_features,
  #         rtol=rtol, atol=atol)


if __name__ == '__main__':
  absltest.main()
