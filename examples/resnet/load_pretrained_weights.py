# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import jax
from flax.core import freeze
from flax.training import train_state
from flax.training import checkpoints

from typing import Any


class TrainState(train_state.TrainState):
  batch_stats: Any


def load_pretrained_weights(state, location):
  chk_state = checkpoints.restore_checkpoint(location, None)

  chk_weights, _ = jax.tree_util.tree_flatten(chk_state['params'])
  _, weight_def = jax.tree_util.tree_flatten(state.params['params'])
  params = jax.tree_util.tree_unflatten(weight_def, chk_weights)

  chk_batchstats, _ = jax.tree_util.tree_flatten(chk_state['batch_stats'])
  _, batchstats_def = jax.tree_util.tree_flatten(state.batch_stats)
  batch_stats = jax.tree_util.tree_unflatten(batchstats_def, chk_batchstats)

  # ml_collections.FrozenConfigDict(
  general_params = {'params': params,
                    'quant_params': state.params['quant_params']}

  return TrainState.create(
      apply_fn=state.apply_fn,
      params=general_params,
      tx=state.tx,
      batch_stats=freeze(batch_stats),
  )
