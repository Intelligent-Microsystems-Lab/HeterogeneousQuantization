# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

"""EfficientNet example.

This script trains a EfficientB0 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

from typing import Any

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax.core import freeze, unfreeze

import jax
from jax import lax
import jax.numpy as jnp

import ml_collections

import optax

Array = jnp.ndarray


def create_model(*, model_cls, num_classes, **kwargs):
  model_dtype = jnp.float32
  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  key, rng = jax.random.split(key, 2)

  @jax.jit
  def init(*args):
    return model.init(*args, rng=rng, train=False)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

  variables = unfreeze(variables)
  if 'quant_params' not in variables:
    variables['quant_params'] = {}
  if 'weight_size' not in variables:
    variables['weight_size'] = {}
  if 'act_size' not in variables:
    variables['act_size'] = {}
  variables = freeze(variables)

  return variables['params'], variables['quant_params'],\
      variables['batch_stats'], variables['weight_size'], variables['act_size']


def cross_entropy_loss(logits, labels, num_classes):
  one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)

  factor = .1
  one_hot_labels *= (1 - factor)
  one_hot_labels += (factor / one_hot_labels.shape[1])

  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels, num_classes, state):
  loss = cross_entropy_loss(logits, labels, num_classes)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  if 'weight_size' in state:
    if state['weight_size'] != {}:
      metrics['weight_size'] = jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(state['weight_size'])[0]))
  if 'act_size' in state:
    if state['act_size'] != {}:
      metrics['act_size_sum'] = jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(state['act_size'])[0]))
      metrics['act_size_max'] = jnp.max(
          jnp.array(jax.tree_util.tree_flatten(state['act_size'])[0]))

  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(config: ml_collections.ConfigDict,
                            base_learning_rate: float,
                            steps_per_epoch: int):
  """Create learning rate schedule."""

  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def train_step(state, batch, rng, learning_rate_fn, num_classes, weight_decay,
               quant_target):
  """Perform a single training step."""
  rng, prng = jax.random.split(rng, 2)

  def loss_fn(params, inputs, targets, quant_params, return_state=True):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn({'params': params,
                                              'quant_params': quant_params,
                                              'batch_stats': state.batch_stats,
                                              'weight_size': state.weight_size,
                                              'act_size': state.act_size},
                                             inputs, rng=prng, mutable=[
        'batch_stats', 'weight_size', 'act_size'],
        rngs={'dropout': rng})
    loss = cross_entropy_loss(logits, targets, num_classes)
    weight_penalty_params = jax.tree_leaves(params)
    weight_l2 = sum([jnp.sum(x ** 2)
                     for x in weight_penalty_params
                     if x.ndim > 1])
    weight_penalty = weight_decay * 0.5 * weight_l2

    # size penalty
    size_penalty = 0.
    if hasattr(quant_target, 'weight_mb'):
      size_penalty += quant_target.weight_penalty * jax.nn.relu(jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(new_model_state['weight_size']
                                               )[0])) - quant_target.weight_mb
      ) ** 2
    if hasattr(quant_target, 'act_size'):
      if quant_target.act_mode == 'sum':
        size_penalty += quant_target.act_penalty * jax.nn.relu(jnp.sum(
            jnp.array(jax.tree_util.tree_flatten(new_model_state['act_size']
                                                 )[0])) - quant_target.act_mb
        ) ** 2
      elif quant_target.act_mode == 'max':
        size_penalty += quant_target.act_penalty * jax.nn.relu(jnp.max(
            jnp.array(jax.tree_util.tree_flatten(new_model_state['act_size']
                                                 )[0])) - quant_target.act_mb
        ) ** 2
      else:
        raise Exception(
            'Unrecongized quant act mode, either sum or max but \
            got: ' + quant_target.act_mode)

    loss = loss + weight_penalty + size_penalty
    if return_state:
      return loss, (new_model_state, logits)
    else:
      return loss

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 3], has_aux=True)
  aux, grads = grad_fn(
      state.params['params'], batch['image'], batch['label'],
      state.params['quant_params'])
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  new_model_state, logits = aux[1]
  metrics = compute_metrics(
      logits, batch['label'], num_classes, new_model_state)
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads={'params': grads[0], 'quant_params': grads[1]},
      batch_stats=new_model_state['batch_stats'],
      weight_size=new_model_state['weight_size'],
      act_size=new_model_state['act_size'])

  return new_state, metrics


def eval_step(state, batch, num_classes):
  variables = {'params': state.params['params'],
               'quant_params': state.params['quant_params'],
               'batch_stats': state.batch_stats,
               'weight_size': state.weight_size, 'act_size': state.act_size, }
  logits, new_state = state.apply_fn(
      variables,
      batch['image'],
      rng=jax.random.PRNGKey(0),
      train=False,
      mutable=['weight_size', 'act_size'])
  return compute_metrics(logits, batch['label'], num_classes, new_state)


class TrainState(train_state.TrainState):
  batch_stats: Any
  weight_size: Any
  act_size: Any


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  """Create initial training state."""

  params, quant_params, batch_stats, weight_size, act_size = initialized(
      rng, image_size, model)
  tx = optax.rmsprop(
      learning_rate=learning_rate_fn,
      decay=0.9,
      momentum=config.momentum,
      eps=0.001,
  )
  quant_params = unfreeze(quant_params)
  quant_params['placeholder'] = 0.
  quant_params = freeze(quant_params)

  state = TrainState.create(
      apply_fn=model.apply,
      params={'params': params, 'quant_params': quant_params},
      tx=tx,
      batch_stats=batch_stats,
      weight_size=weight_size,
      act_size=act_size,
  )
  return state
