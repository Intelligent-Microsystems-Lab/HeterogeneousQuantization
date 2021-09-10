# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

# Test data obtained from


from absl.testing import absltest
import functools

import jax
import jax.numpy as jnp
import numpy as np

from flax.core import freeze, unfreeze

from utils import sse_loss
from pc_modular import DensePC, PC_NN

import ml_collections


class UnitTests(absltest.TestCase):
  def test_pc_out_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/dataset_x.npy")
    y = np.load("unit_test/dataset_y.npy")

    l1w = np.load("unit_test/layer_0_weights.npy")
    l2w = np.load("unit_test/layer_1_weights.npy")

    out_ref = np.load("unit_test/out_train.npy")

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

    rng, trng = jax.random.split(rng, 2)
    variables = nn_unit_test.init(trng, train_x)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.array(l1w)
    params["layers_1"]["kernel"] = jnp.array(l2w)
    params = freeze(params)

    out, state = nn_unit_test.apply(
        {"params": params, **state}, train_x, mutable=list(state.keys())
    )

    np.testing.assert_almost_equal(out, out_ref, decimal=6)

  def test_pc_err_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/dataset_x.npy")
    y = np.load("unit_test/dataset_y.npy")

    l1w = np.load("unit_test/layer_0_weights.npy")
    l2w = np.load("unit_test/layer_1_weights.npy")

    err0_ref = np.load("unit_test/pred0_train.npy")
    err1_ref = np.load("unit_test/pred1_train.npy")

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

    rng, trng = jax.random.split(rng, 2)
    variables = nn_unit_test.init(trng, train_x)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.array(l1w)
    params["layers_1"]["kernel"] = jnp.array(l2w)
    params = freeze(params)

    out, state = nn_unit_test.apply(
        {"params": params, **state}, train_x, mutable=list(state.keys())
    )

    err, state = nn_unit_test.apply(
        {"params": params, **state},
        train_y,
        out,
        mutable=list(state.keys()),
        method=PC_NN.inference,
    )

    np.testing.assert_almost_equal(err["layers_0"], err0_ref, decimal=5)
    np.testing.assert_almost_equal(err["layers_1"], err1_ref, decimal=5)

  def test_pc_dw_equality(self):
    cfg = ml_collections.ConfigDict()
    cfg.infer_lr = 0.2
    cfg.infer_steps = 100

    rng = jax.random.PRNGKey(0)

    x = np.load("unit_test/dataset_x.npy")
    y = np.load("unit_test/dataset_y.npy")

    l1w = np.load("unit_test/layer_0_weights.npy")
    l2w = np.load("unit_test/layer_1_weights.npy")

    dw0_ref = np.load("unit_test/dw0_train.npy")
    dw1_ref = np.load("unit_test/dw1_train.npy")

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

    rng, trng = jax.random.split(rng, 2)
    variables = nn_unit_test.init(trng, train_x)
    state, params = variables.pop("params")

    params = unfreeze(params)
    params["layers_0"]["kernel"] = jnp.array(l1w)
    params["layers_1"]["kernel"] = jnp.array(l2w)
    params = freeze(params)

    grads, state = nn_unit_test.apply(
        {"params": params, **state},
        train_x,
        train_y,
        mutable=list(state.keys()),
        method=PC_NN.grads,
    )

    np.testing.assert_almost_equal(
        grads["layers_0"]["kernel"], dw0_ref, decimal=5
    )
    np.testing.assert_almost_equal(
        grads["layers_1"]["kernel"], dw1_ref, decimal=3
    )


if __name__ == "__main__":
  absltest.main()
