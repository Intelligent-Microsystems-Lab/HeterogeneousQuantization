import tensorflow as tf
from flax.core import freeze, unfreeze
from flax.training import train_state
import jax.numpy as jnp

from typing import Any


class TrainState(train_state.TrainState):
  batch_stats: Any


def test_shapes(shape1, shape2, name):
  assert shape1 == shape2, (
      "Checkpoint corrupt, shapes incompatible: "
      + str(shape1)
      + " vs "
      + str(shape2)
      + " at layer "
      + name
  )


def load_pretrained_weights(state, location):
  tf_vars = tf.train.list_variables(location)
  params = unfreeze(state.params)
  batch_stats = unfreeze(state.batch_stats)

  for name, shape in tf_vars:
    list_components = name.split("/")
    if len(list_components) == 1:
      # Entry for step - ignored so far.
      continue
    if len(list_components) == 4:
      # net, layer, op, param = list_components
      continue
    elif len(list_components) == 5:
      # Exponential Moving Average.
      net, layer, op, param, _ = list_components
      # continue
    else:
      raise Exception("Checkpoint corrupt: " + location)

    if "blocks" in layer:

      if "depthwise" in op:
        loaded_transposed = jnp.moveaxis(
            jnp.array(tf.train.load_variable(location, name)),
            (0, 1, 2, 3),
            (0, 1, 3, 2),
        )
        test_shapes(
            loaded_transposed.shape,
            params["MBConvBlock_" + layer.split("_")[-1]][
                "depthwise_conv2d"
            ]["kernel"].shape,
            name,
        )
        params["MBConvBlock_" + layer.split("_")[-1]][
            "depthwise_conv2d"
        ]["kernel"] = loaded_transposed
        continue

      if "conv2d" in op:
        flax_op_num = (
            "0" if len(op.split("_")) == 1 else op.split("_")[-1]
        )
        test_shapes(
            shape,
            list(
                params["MBConvBlock_" + layer.split("_")[-1]][
                    "QuantConv_" + flax_op_num
                ]["kernel"].shape
            ),
            name,
        )
        params["MBConvBlock_" + layer.split("_")[-1]][
            "QuantConv_" + flax_op_num
        ]["kernel"] = jnp.array(tf.train.load_variable(location, name))
        continue

      if "tpu_batch_normalization" in op:
        num_bn = "0" if len(op.split("_")) == 3 else op.split("_")[-1]
        if param == "beta":
          test_shapes(
              shape,
              list(
                  params["MBConvBlock_" + layer.split("_")[-1]][
                      "BatchNorm_" + num_bn
                  ]["bias"].shape
              ),
              name,
          )
          params["MBConvBlock_" + layer.split("_")[-1]][
              "BatchNorm_" + num_bn
          ]["bias"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue
        if param == "gamma":
          test_shapes(
              shape,
              list(
                  params["MBConvBlock_" + layer.split("_")[-1]][
                      "BatchNorm_" + num_bn
                  ]["scale"].shape
              ),
              name,
          )
          params["MBConvBlock_" + layer.split("_")[-1]][
              "BatchNorm_" + num_bn
          ]["scale"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue
        if param == "moving_mean":
          test_shapes(
              shape,
              list(
                  batch_stats["MBConvBlock_" + layer.split("_")[-1]][
                      "BatchNorm_" + num_bn
                  ]["mean"].shape
              ),
              name,
          )
          batch_stats["MBConvBlock_" + layer.split("_")[-1]][
              "BatchNorm_" + num_bn
          ]["mean"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue
        if param == "moving_variance":
          test_shapes(
              shape,
              list(
                  batch_stats["MBConvBlock_" + layer.split("_")[-1]][
                      "BatchNorm_" + num_bn
                  ]["var"].shape
              ),
              name,
          )
          batch_stats["MBConvBlock_" + layer.split("_")[-1]][
              "BatchNorm_" + num_bn
          ]["var"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue

    if ("stem" in layer) or ("head" in layer):
      if op == "dense":
        test_shapes(shape, list(params["QuantDense_0"][param].shape), name)
        params["QuantDense_0"][param] = jnp.array(
            tf.train.load_variable(location, name)
        )
        continue
      if op == "conv2d":
        test_shapes(
            shape, list(params[layer + "_conv"]["kernel"].shape), name
        )
        params[layer + "_conv"]["kernel"] = jnp.array(
            tf.train.load_variable(location, name)
        )
        continue
      if op == "tpu_batch_normalization":
        if param == "beta":
          test_shapes(
              shape, list(params[layer + "_bn"]["bias"].shape), name
          )
          params[layer + "_bn"]["bias"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue
        if param == "gamma":
          test_shapes(
              shape, list(params[layer + "_bn"]["scale"].shape), name
          )
          params[layer + "_bn"]["scale"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue
        if param == "moving_mean":
          test_shapes(
              shape,
              list(batch_stats[layer + "_bn"]["mean"].shape),
              name,
          )
          batch_stats[layer + "_bn"]["mean"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue
        if param == "moving_variance":
          test_shapes(
              shape,
              list(batch_stats[layer + "_bn"]["var"].shape),
              name,
          )
          batch_stats[layer + "_bn"]["var"] = jnp.array(
              tf.train.load_variable(location, name)
          )
          continue

  return TrainState.create(
      apply_fn=state.apply_fn,
      params=freeze(params),
      tx=state.tx,
      batch_stats=freeze(batch_stats),
  )
