# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

# Test data obtained from
# https://github.com/BerenMillidge/PredictiveCodingBackprop
# Find modified files to obtain test data in unit_test sub directory.

from absl.testing import absltest
from absl.testing import parameterized
import functools

import jax
import jax.numpy as jnp
import numpy as np

from flax.core import freeze, unfreeze

from jax.nn import initializers

from utils import sse_loss
from pc_modular import DensePC, PC_NN, ConvolutionalPC

import ml_collections


# from jax.config import config
# config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)


def dense_act_noise_data():
  return (
      dict(
          testcase_name="noise_act_01",
          examples=1000,
          inp_channels=2000,
          out_channels=3000,
          noise=0.01,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_act_05",
          examples=1000,
          inp_channels=3000,
          out_channels=2000,
          noise=0.05,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_act_10",
          examples=2000,
          inp_channels=4000,
          out_channels=2000,
          noise=0.1,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_act_20",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=5e-2,
      ),
  )


def dense_weight_noise_data():
  return (
      dict(
          testcase_name="noise_weight_01",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.01,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_weight_05",
          examples=1000,
          inp_channels=3000,
          out_channels=2000,
          noise=0.05,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_weight_10",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.1,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_weight_20",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=5e-2,
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
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_err_inpt_05",
          examples=1000,
          inp_channels=3000,
          out_channels=2000,
          noise=0.05,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_err_inpt_10",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.1,
          numerical_tolerance=5e-2,
      ),
      dict(
          testcase_name="noise_err_inpt_20",
          examples=1000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=5e-2,
      ),
  )


def dense_err_weight_noise_data():
  return (
      dict(
          testcase_name="noise_weight_noise_01",
          examples=10000,
          inp_channels=1000,
          out_channels=2000,
          noise=0.01,
          numerical_tolerance=1e-3,
      ),
      dict(
          testcase_name="noise_weight_noise_05",
          examples=10000,
          inp_channels=3000,
          out_channels=2000,
          noise=0.05,
          numerical_tolerance=1e-3,
      ),
      dict(
          testcase_name="noise_weight_noise_10",
          examples=10000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.1,
          numerical_tolerance=1e-3,
      ),
      dict(
          testcase_name="noise_weight_noise_20",
          examples=10000,
          inp_channels=2000,
          out_channels=2000,
          noise=0.2,
          numerical_tolerance=1e-3,
      ),
  )


class UnitTests(parameterized.TestCase):
  def test_pc_fc_out_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100
    cfg.weight_noise = 0.0
    cfg.act_noise = 0.0
    cfg.err_inpt_noise = 0.0
    cfg.err_weight_noise = 0.0

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/unit_test_fc/dataset_x.npy")
    y = np.load("unit_test/unit_test_fc/dataset_y.npy")

    l1w = np.load("unit_test/unit_test_fc/layer_0_weights.npy")
    l2w = np.load("unit_test/unit_test_fc/layer_1_weights.npy")

    out_ref = np.load("unit_test/unit_test_fc/out_train.npy")

    train_x = x[:128]
    train_y = y[:128]

    test_x = x[128:]
    test_y = y[128:]

    class test_pc_nn(PC_NN):
      def setup(self):
        self.layers = [
            DensePC(100, config=cfg),
            DensePC(10, config=cfg),
        ]

    nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

    rng, trng, subkey1, subkey2 = jax.random.split(rng, 4)
    variables = nn_unit_test.init(trng, train_x, subkey2)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.array(l1w)
    params["layers_1"]["kernel"] = jnp.array(l2w)
    params = freeze(params)

    out, state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        subkey1,
        mutable=list(state.keys()),
    )

    np.testing.assert_almost_equal(out, out_ref, decimal=6)

  def test_pc_fc_err_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100
    cfg.weight_noise = 0.0
    cfg.act_noise = 0.0
    cfg.err_inpt_noise = 0.0
    cfg.err_weight_noise = 0.0

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/unit_test_fc/dataset_x.npy")
    y = np.load("unit_test/unit_test_fc/dataset_y.npy")

    l1w = np.load("unit_test/unit_test_fc/layer_0_weights.npy")
    l2w = np.load("unit_test/unit_test_fc/layer_1_weights.npy")

    err0_ref = np.load("unit_test/unit_test_fc/pred0_train.npy")
    err1_ref = np.load("unit_test/unit_test_fc/pred1_train.npy")

    train_x = x[:128]
    train_y = y[:128]

    test_x = x[128:]
    test_y = y[128:]

    class test_pc_nn(PC_NN):
      def setup(self):
        self.layers = [
            DensePC(100, config=cfg),
            DensePC(10, config=cfg),
        ]

    nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

    rng, trng, subkey1, subkey2, subkey3 = jax.random.split(rng, 5)
    variables = nn_unit_test.init(trng, train_x, subkey3)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.array(l1w)
    params["layers_1"]["kernel"] = jnp.array(l2w)
    params = freeze(params)

    (out, err_init), state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        subkey1,
        True,
        mutable=list(state.keys()),
    )

    err, state = nn_unit_test.apply(
        {"params": params, **state},
        train_y,
        out,
        subkey2,
        err_init,
        mutable=list(state.keys()),
        method=PC_NN.inference,
    )

    np.testing.assert_almost_equal(err["layers_0"], err0_ref, decimal=5)
    np.testing.assert_almost_equal(err["layers_1"], err1_ref, decimal=5)

  # def test_pc_fc_bp_equality(self):
  #   cfg = ml_collections.ConfigDict()
  #   cfg.infer_lr = 0.2
  #   cfg.infer_steps = 100

  #   rng = jax.random.PRNGKey(0)

  #   train_x = x[:128]
  #   train_y = y[:128]

  #   test_x = x[128:]
  #   test_y = y[128:]

  #   class test_pc_nn(PC_NN):
  #     def setup(self):
  #       self.layers = [
  #           DensePC(100, config=cfg),
  #           DensePC(10, config=cfg),
  #       ]

  #   nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

  #   rng, trng = jax.random.split(rng, 2)
  #   variables = nn_unit_test.init(trng, train_x)
  #   state, params = variables.pop("params")

  #   params = unfreeze(params)
  #   params["layers_0"]["kernel"] = jnp.array(l1w)
  #   params["layers_1"]["kernel"] = jnp.array(l2w)
  #   params = freeze(params)

  #   grads, state = nn_unit_test.apply(
  #       {"params": params, **state},
  #       train_x,
  #       train_y,
  #       mutable=list(state.keys()),
  #       method=PC_NN.grads,
  #   )

  #   np.testing.assert_almost_equal(
  #       grads["layers_0"]["kernel"], dw0_ref, decimal=5
  #   )
  #   np.testing.assert_almost_equal(
  #       grads["layers_1"]["kernel"], dw1_ref, decimal=3
  #   )

  def test_pc_fc_dw_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100
    cfg.weight_noise = 0.0
    cfg.act_noise = 0.0
    cfg.err_inpt_noise = 0.0
    cfg.err_weight_noise = 0.0

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/unit_test_fc/dataset_x.npy")
    y = np.load("unit_test/unit_test_fc/dataset_y.npy")

    l1w = np.load("unit_test/unit_test_fc/layer_0_weights.npy")
    l2w = np.load("unit_test/unit_test_fc/layer_1_weights.npy")

    dw0_ref = np.load("unit_test/unit_test_fc/dw0_train.npy")
    dw1_ref = np.load("unit_test/unit_test_fc/dw1_train.npy")

    train_x = x[:128]
    train_y = y[:128]

    test_x = x[128:]
    test_y = y[128:]

    class test_pc_nn(PC_NN):
      def setup(self):
        self.layers = [
            DensePC(100, config=cfg),
            DensePC(10, config=cfg),
        ]

    nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

    rng, trng, subkey1, subkey2 = jax.random.split(rng, 4)
    variables = nn_unit_test.init(trng, train_x, subkey2)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.array(l1w)
    params["layers_1"]["kernel"] = jnp.array(l2w)
    params = freeze(params)

    (grads, _), state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        train_y,
        subkey1,
        mutable=list(state.keys()),
        method=PC_NN.grads,
    )

    np.testing.assert_almost_equal(
        grads["layers_0"]["kernel"], -1 * dw0_ref, decimal=4
    )
    np.testing.assert_almost_equal(
        grads["layers_1"]["kernel"], -1 * dw1_ref, decimal=3
    )

  # Conv Test
  def test_pc_conv_out_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100
    cfg.weight_noise = 0.0
    cfg.act_noise = 0.0
    cfg.err_inpt_noise = 0.0
    cfg.err_weight_noise = 0.0

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/unit_test_conv/dataset_x.npy")
    y = np.load("unit_test/unit_test_conv/dataset_y.npy")

    l1w = np.load("unit_test/unit_test_conv/layer_0_weights.npy")
    l2w = np.load("unit_test/unit_test_conv/layer_1_weights.npy")

    out_ref = np.load("unit_test/unit_test_conv/out_train.npy")

    train_x = jnp.moveaxis(x[:128], (0, 1, 2, 3), (0, 3, 2, 1))
    train_y = jnp.moveaxis(y[:128], (0, 1, 2, 3), (0, 3, 2, 1))

    test_x = jnp.moveaxis(x[128:], (0, 1, 2, 3), (0, 3, 2, 1))
    test_y = jnp.moveaxis(y[128:], (0, 1, 2, 3), (0, 3, 2, 1))

    class test_pc_nn(PC_NN):
      def setup(self):
        self.layers = [
            ConvolutionalPC(
                features=6,
                kernel_size=(5, 5),
                padding="VALID",
                config=cfg,
            ),
            ConvolutionalPC(
                features=12,
                kernel_size=(5, 5),
                padding="VALID",
                config=cfg,
            ),
        ]

    nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

    rng, trng, subkey1, subkey2, subkey3 = jax.random.split(rng, 5)
    variables = nn_unit_test.init(trng, train_x, subkey3)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.transpose(l1w)
    params["layers_1"]["kernel"] = jnp.transpose(l2w)
    params = freeze(params)

    out, state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        subkey1,
        mutable=list(state.keys()),
    )
    np.testing.assert_almost_equal(
        np.array(out),
        np.array(jnp.moveaxis(out_ref, (0, 1, 2, 3), (0, 3, 2, 1))),
        decimal=6,
    )

  def test_pc_conv_err_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100
    cfg.weight_noise = 0.0
    cfg.act_noise = 0.0
    cfg.err_inpt_noise = 0.0
    cfg.err_weight_noise = 0.0

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/unit_test_conv/dataset_x.npy")
    y = np.load("unit_test/unit_test_conv/dataset_y.npy")

    l1w = np.load("unit_test/unit_test_conv/layer_0_weights.npy")
    l2w = np.load("unit_test/unit_test_conv/layer_1_weights.npy")

    err0_ref = np.load("unit_test/unit_test_conv/pred0_train.npy")
    err1_ref = np.load("unit_test/unit_test_conv/pred1_train.npy")

    train_x = jnp.moveaxis(x[:128], (0, 1, 2, 3), (0, 3, 2, 1))
    train_y = jnp.moveaxis(y[:128], (0, 1, 2, 3), (0, 3, 2, 1)).astype(
        jnp.int8
    )

    test_x = jnp.moveaxis(x[128:], (0, 1, 2, 3), (0, 3, 2, 1))
    test_y = jnp.moveaxis(y[128:], (0, 1, 2, 3), (0, 3, 2, 1)).astype(
        jnp.int8
    )

    class test_pc_nn(PC_NN):
      def setup(self):
        self.layers = [
            ConvolutionalPC(
                features=6,
                kernel_size=(5, 5),
                padding="VALID",
                config=cfg,
            ),
            ConvolutionalPC(
                features=12,
                kernel_size=(5, 5),
                padding="VALID",
                config=cfg,
            ),
        ]

    nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

    rng, trng, subkey1, subkey2, subkey3 = jax.random.split(rng, 5)
    variables = nn_unit_test.init(trng, train_x, subkey3)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.transpose(l1w)
    params["layers_1"]["kernel"] = jnp.transpose(l2w)
    params = freeze(params)

    (out, err_init), state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        subkey2,
        True,
        mutable=list(state.keys()),
    )

    err, state = nn_unit_test.apply(
        {"params": params, **state},
        train_y,
        out,
        subkey1,
        err_init,
        mutable=list(state.keys()),
        method=PC_NN.inference,
    )

    np.testing.assert_almost_equal(
        np.array(err["layers_0"]),
        np.array(jnp.moveaxis(err0_ref, (0, 1, 2, 3), (0, 3, 2, 1))),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        np.array(err["layers_1"]),
        np.array(jnp.moveaxis(err1_ref, (0, 1, 2, 3), (0, 3, 2, 1))),
        decimal=6,
    )

  def test_pc_conv_dw_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100
    cfg.weight_noise = 0.0
    cfg.act_noise = 0.0
    cfg.err_inpt_noise = 0.0
    cfg.err_weight_noise = 0.0

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/unit_test_conv/dataset_x.npy")
    y = np.load("unit_test/unit_test_conv/dataset_y.npy")

    l1w = np.load("unit_test/unit_test_conv/layer_0_weights.npy")
    l2w = np.load("unit_test/unit_test_conv/layer_1_weights.npy")

    dw0_ref = np.load("unit_test/unit_test_conv/dw0_train.npy")
    dw1_ref = np.load("unit_test/unit_test_conv/dw1_train.npy")

    train_x = jnp.moveaxis(x[:128], (0, 1, 2, 3), (0, 3, 2, 1))
    train_y = jnp.moveaxis(y[:128], (0, 1, 2, 3), (0, 3, 2, 1)).astype(
        jnp.int8
    )

    test_x = jnp.moveaxis(x[128:], (0, 1, 2, 3), (0, 3, 2, 1))
    test_y = jnp.moveaxis(y[128:], (0, 1, 2, 3), (0, 3, 2, 1)).astype(
        jnp.int8
    )

    class test_pc_nn(PC_NN):
      def setup(self):
        self.layers = [
            ConvolutionalPC(
                features=6,
                kernel_size=(5, 5),
                padding="VALID",
                config=cfg,
            ),
            ConvolutionalPC(
                features=12,
                kernel_size=(5, 5),
                padding="VALID",
                config=cfg,
            ),
        ]

    nn_unit_test = test_pc_nn(config=cfg, loss_fn=sse_loss)

    rng, trng, subkey1, subkey2 = jax.random.split(rng, 4)
    variables = nn_unit_test.init(trng, train_x, subkey2)
    state, params = variables.pop("params")
    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.transpose(l1w)
    params["layers_1"]["kernel"] = jnp.transpose(l2w)
    params = freeze(params)
    (grads, _), state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        train_y,
        subkey1,
        mutable=list(state.keys()),
        method=PC_NN.grads,
    )
    np.testing.assert_almost_equal(
        np.array(grads["layers_0"]["kernel"]),
        -1 * np.array((dw0_ref).transpose()),
        decimal=2,
    )
    np.testing.assert_almost_equal(
        np.array(grads["layers_1"]["kernel"]),
        -1 * np.array((dw1_ref).transpose()),
        decimal=2,
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
            "err_weight_nois": 0.0,
            "infer_lr": 1,
        }
    )

    key = jax.random.PRNGKey(34835972)
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)

    data = jnp.ones((examples, inp_channels))

    test_dense = DensePC(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)
    state, params = variables.pop("params")
    out_d, state_out = test_dense.apply(
        {"params": params, **state}, data, rng3, mutable=list(state.keys()),)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(out_d),
        inp_channels,
        rtol=numerical_tolerance,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(out_d),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * inp_channels),
        rtol=numerical_tolerance,
    )

    grads_wrt_weights, state_out2 = test_dense.apply(
        {"params": params, **state}, jnp.ones_like(out_d), rng4, mutable=list(state.keys()), method=test_dense.grads,)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_weights["kernel"]),
        examples,
        rtol=numerical_tolerance,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(grads_wrt_weights["kernel"]),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * examples),
        rtol=numerical_tolerance,
    )

    # state = unfreeze(state)
    # state['pc']['out'] = jnp.ones_like(state['pc']['out'])
    # state = freeze(state)

    # (pe, _), _ = test_dense.apply({"params": params, **state}, jnp.ones_like(
    #     out_d), data, rng4, mutable=list(state.keys()), method=test_dense.infer,)

    # # test for mean
    # np.testing.assert_allclose(
    #     1+jnp.mean(pe),
    #     1+0,
    #     rtol=numerical_tolerance,
    # )

    # # test for variance
    # np.testing.assert_allclose(
    #     jnp.std(pe),
    #     np.sqrt((2*noise)**2/12),
    #     rtol=numerical_tolerance,
    # )

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
            "err_weight_nois": 0.0,
            "infer_lr": 1,
        }
    )

    key = jax.random.PRNGKey(34835972)
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)

    data = jnp.ones((examples, inp_channels))

    test_dense = DensePC(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)
    state, params = variables.pop("params")
    out_d, state_out = test_dense.apply(
        {"params": params, **state}, data, rng3, mutable=list(state.keys()),)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(out_d),
        inp_channels,
        rtol=numerical_tolerance,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(out_d),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * inp_channels),
        rtol=numerical_tolerance,
    )

    state = unfreeze(state)
    state['pc']['value'] = jnp.zeros_like(state['pc']['value'])
    state = freeze(state)

    (_, grads_wrt_inpt), _ = test_dense.apply({"params": params, **state}, jnp.ones_like(
        out_d), jnp.zeros_like(data), rng4, mutable=list(state.keys()), method=test_dense.infer,)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_inpt),
        out_channels,
        rtol=numerical_tolerance,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(grads_wrt_inpt),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * out_channels),
        rtol=numerical_tolerance,
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
            "err_weight_nois": 0.0,
            "infer_lr": 1,
        }
    )

    key = jax.random.PRNGKey(34835972)
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)

    data = jnp.ones((examples, inp_channels))

    test_dense = DensePC(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)
    state, params = variables.pop("params")
    out_d, state_out = test_dense.apply(
        {"params": params, **state}, data, rng3, mutable=list(state.keys()),)

    state = unfreeze(state)
    state['pc']['value'] = jnp.zeros_like(state['pc']['value'])
    state = freeze(state)

    (_, grads_wrt_inpt), _ = test_dense.apply({"params": params, **state}, jnp.ones_like(
        out_d), jnp.zeros_like(data), rng4, mutable=list(state.keys()), method=test_dense.infer,)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_inpt),
        out_channels,
        rtol=numerical_tolerance,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(grads_wrt_inpt),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * out_channels),
        rtol=numerical_tolerance,
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
            "infer_lr": 1,
        }
    )

    key = jax.random.PRNGKey(34835972)
    rng1, rng2, rng3, rng4 = jax.random.split(key, 4)

    data = jnp.ones((examples, inp_channels))

    test_dense = DensePC(
        features=out_channels,
        kernel_init=initializers.ones,
        config=config,
    )

    variables = test_dense.init(rng1, data, rng2)
    state, params = variables.pop("params")
    out_d, state_out = test_dense.apply(
        {"params": params, **state}, data, rng3, mutable=list(state.keys()),)

    grads_wrt_weights, state_out2 = test_dense.apply(
        {"params": params, **state}, jnp.ones_like(out_d), rng4, mutable=list(state.keys()), method=test_dense.grads,)

    # test for mean
    np.testing.assert_allclose(
        jnp.mean(grads_wrt_weights["kernel"]),
        examples,
        rtol=numerical_tolerance,
    )

    # test for variance
    np.testing.assert_allclose(
        jnp.std(grads_wrt_weights["kernel"]),
        jnp.sqrt((1 / 12 * (noise * 2) ** 2) * examples),
        rtol=numerical_tolerance,
    )


if __name__ == "__main__":
  absltest.main()
