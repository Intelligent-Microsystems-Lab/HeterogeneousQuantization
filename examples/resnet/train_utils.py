# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples

from typing import Any

import input_pipeline

from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax.core import freeze, unfreeze

import jax
from jax import lax

import jax.numpy as jnp

import ml_collections

import optax


def create_model(*, model_cls, **kwargs):
  model_dtype = jnp.float32
  return model_cls(num_classes=kwargs['config'].num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  key, rng = jax.random.split(key, 2)

  @jax.jit
  def init(*args):
    return model.init(*args, train=False)
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


def cross_entropy_loss(logits, labels):
  one_hot_labels = common_utils.onehot(labels, num_classes=logits.shape[1])
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels, state, size_div):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  if 'weight_size' in state:
    if state['weight_size'] != {}:
      metrics['weight_size'] = jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(state['weight_size'])[0]))/size_div
  if 'act_size' in state:
    if state['act_size'] != {}:
      metrics['act_size_sum'] = jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(state['act_size'])[0]))/size_div
      metrics['act_size_max'] = jnp.max(
          jnp.array(jax.tree_util.tree_flatten(state['act_size'])[0]))/size_div

  # metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int):
  """Create learning rate schedule."""
  if config.lr_boundaries_scale is not None:
    schedule_fn = optax.piecewise_constant_schedule(config.learning_rate, {int(k)*steps_per_epoch:v for k,v in config.lr_boundaries_scale.items()})
  else: 
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


def train_step(state, batch, learning_rate_fn, weight_decay, quant_target):
  """Perform a single training step."""
  def loss_fn(params, quant_params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
        {'params': params, 'quant_params': quant_params,
         'batch_stats': state.batch_stats, 'weight_size': state.weight_size,
         'act_size': state.act_size},
        batch['image'], mutable=['batch_stats', 'weight_size', 'act_size'])
    loss = cross_entropy_loss(logits, batch['label'])
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
                                               )[0]))/quant_target.size_div - quant_target.weight_mb
      ) ** 2
    if hasattr(quant_target, 'act_size'):
      if quant_target.act_mode == 'sum':
        size_penalty += quant_target.act_penalty * jax.nn.relu(jnp.sum(
            jnp.array(jax.tree_util.tree_flatten(new_model_state['act_size']
                                                 )[0]))/quant_target.size_div - quant_target.act_mb
        ) ** 2
      elif quant_target.act_mode == 'max':
        size_penalty += quant_target.act_penalty * jax.nn.relu(jnp.max(
            jnp.array(jax.tree_util.tree_flatten(new_model_state['act_size']
                                                 )[0]))/quant_target.size_div - quant_target.act_mb
        ) ** 2
      else:
        raise Exception(
            'Unrecongized quant act mode, either sum or max but got: '
            + quant_target.act_mode)

    loss = loss + weight_penalty + size_penalty
    return loss, (new_model_state, logits)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)
  aux, grads = grad_fn(state.params['params'], state.params['quant_params'])
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  # grads = lax.pmean(grads, axis_name='batch')

  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'], new_model_state, quant_target.size_div)
  metrics['learning_rate'] = lr
  new_state = state.apply_gradients(
      grads={'params': grads[0], 'quant_params': grads[1]},
      batch_stats=new_model_state['batch_stats'],
      weight_size=new_model_state['weight_size'],
      act_size=new_model_state['act_size'])

  return new_state, metrics


def eval_step(state, batch, size_div):
  variables = {'params': state.params['params'],
               'quant_params': state.params['quant_params'],
               'batch_stats': state.batch_stats,
               'weight_size': state.weight_size, 'act_size': state.act_size, }
  logits, new_state = state.apply_fn(
      variables,
      batch['image'],
      train=False,
      mutable=['weight_size', 'act_size'])
  return compute_metrics(logits, batch['label'], new_state, size_div)


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


# def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
#                       cache):
#   ds = input_pipeline.create_split(
#       dataset_builder, batch_size, image_size=image_size, dtype=dtype,
#       train=train, cache=cache)
#   it = map(prepare_tf_data, ds)
#   it = jax_utils.prefetch_to_device(it, 2)
#   return it


def create_input_iter(
    dataset_builder, batch_size, image_size, dtype, train, cache
):
  if "imagenet" in dataset_builder.name:
    ds = input_pipeline.create_split(
        dataset_builder,
        batch_size,
        image_size=image_size,
        dtype=dtype,
        train=train,
        cache=cache,
    )
  else:
    ds = input_pipeline.create_split_cifar10(
        dataset_builder,
        batch_size,
        image_size=image_size,
        dtype=dtype,
        train=train,
        cache=cache,
    )
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


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
  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=config.nesterov,
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
