# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

"""Tests for MobileNetV2."""

from absl.testing import absltest
from absl import logging

import importlib

import jax
from jax import numpy as jnp
import numpy as np
import optax

from PIL import Image
from torchvision import transforms

from mobilenetv2.configs import mobilenetv2_fp32 as default_lib

import mobilenetv2.models as models
from mobilenetv2.mobilenetv2_load_pretrained_weights import (
    mobilenetv2_load_pretrained_weights
)

from train_utils import TrainState  # noqa: E402

jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_disable_most_optimizations', True)


class MobileNetV2Test(absltest.TestCase):
  """Test cases for ResNet v1 model definition."""

  def test_mobilenetv2_model(self):
    """Tests MobileNetV2 model definition and output (variables)."""
    rng = jax.random.PRNGKey(0)
    config = default_lib.get_config()
    model_def = models.MobileNetV2_100(
        num_classes=1000, dtype=jnp.bfloat16, config=config)
    rng, prng = jax.random.split(rng, 2)
    variables = model_def.init(
        rng, jnp.ones((8, 224, 224, 3), jnp.bfloat16), rng=prng, train=False)

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

  def test_mobilenetv2_inference(self):

    # get correct config
    config_module = importlib.import_module(
        'configs.mobilenetv2_fp32')
    cfg = config_module.get_config()
    cfg.pretrained = '../../pretrained_mobilenetv2/mobilenet_v2-b0353104.pth'

    # initialize network
    rng = jax.random.PRNGKey(0)
    rng, prng1, prng2 = jax.random.split(rng, 3)
    model_cls = getattr(models, 'MobileNetV2_100')
    model_def = model_cls(num_classes=1000, dtype=jnp.float32, config=cfg)
    variables = model_def.init(
        prng1, jnp.ones((8, 224, 224, 3), jnp.float32),
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
        weight_size={}, act_size={}, quant_config={})
    state = mobilenetv2_load_pretrained_weights(state, cfg.pretrained)

    # load inpt
    input_image = Image.open('../../unit_tests/mobilnetv2_unit_test/dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = jnp.moveaxis(
        jnp.array(input_batch), (0, 1, 2, 3), (0, 3, 1, 2))

    # run inference
    rng, prng = jax.random.split(rng, 2)
    _, state = model_def.apply({'params': state.params['params'],
                                'quant_params': state.params['quant_params'],
                                'batch_stats': state.batch_stats,
                                'weight_size': state.weight_size,
                                'act_size': state.act_size}, input_batch,
                               mutable=['intermediates', 'batch_stats',
                                        'weight_size', 'act_size'],
                               rng=prng,
                               train=False)

    rtol = 1e-7
    atol = 1e-4
    # testing for equality
    logging.info('Check stem.')
    pytorch_stem = np.load('../../unit_tests/mobilnetv2_unit_test/stem.npy')
    pytorch_stem = jnp.moveaxis(
        jnp.array(pytorch_stem), (0, 1, 2, 3), (0, 3, 1, 2))
    np.testing.assert_allclose(
        state['intermediates']['stem'][0], pytorch_stem, rtol=rtol, atol=atol)

    for i in range(17):
      logging.info('Check feature: ' + str(i))
      pytorch_features = np.load(
          '../../unit_tests/mobilnetv2_unit_test/features_' + str(i + 1)
          + '.npy')
      pytorch_features = jnp.moveaxis(
          jnp.array(pytorch_features), (0, 1, 2, 3), (0, 3, 1, 2))
      np.testing.assert_allclose(
          state['intermediates']['features_' + str(i)][0], pytorch_features,
          rtol=rtol, atol=atol)

    logging.info('Check head.')
    pytorch_head = np.load('../../unit_tests/mobilnetv2_unit_test/head.npy')
    pytorch_head = jnp.moveaxis(
        jnp.array(pytorch_head), (0, 1, 2, 3), (0, 3, 1, 2))
    np.testing.assert_allclose(
        state['intermediates']['head'][0], pytorch_head, rtol=rtol, atol=atol)


if __name__ == '__main__':
  absltest.main()
