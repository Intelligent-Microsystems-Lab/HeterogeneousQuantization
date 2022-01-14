# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/tree/main/examples


from typing import Any

from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
from flax.core import freeze, unfreeze

import jax
from jax import lax
import flax
import jax.numpy as jnp

import ml_collections

import tree
import optax

Array = jnp.ndarray


@jax.custom_vjp
def max_custom_grad(x):
  return jnp.max(x)


def max_custom_grad_fwd(x):
  return max_custom_grad(x), (x,)


def max_custom_grad_bwd(res, g):
  x, = res
  mask = jnp.where(x == jnp.max(x), 1, 0)
  return (g * mask,)


max_custom_grad.defvjp(max_custom_grad_fwd, max_custom_grad_bwd)


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
  if 'quant_config' not in variables:
    variables['quant_config'] = {}
  variables = freeze(variables)

  return variables['params'], variables['quant_params'], \
      variables['batch_stats'], variables['weight_size'], \
      variables['act_size'], variables['quant_config']


def cross_entropy_loss(logits, labels, smoothing):
  one_hot_labels = common_utils.onehot(labels, num_classes=logits.shape[1])

  factor = smoothing
  one_hot_labels *= (1 - factor)
  one_hot_labels += (factor / one_hot_labels.shape[1])

  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return xentropy


def compute_metrics(logits, labels, state, size_div, smoothing):
  loss = cross_entropy_loss(logits, labels, smoothing)
  accuracy = (jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  if 'weight_size' in state:
    if state['weight_size'] != {}:
      metrics['weight_size'] = jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(state['weight_size'])[0])
      ) / size_div
  if 'act_size' in state:
    if state['act_size'] != {}:
      metrics['act_size_sum'] = jnp.sum(
          jnp.array(jax.tree_util.tree_flatten(state['act_size'])[0])
      ) / size_div
      metrics['act_size_max'] = jnp.max(
          jnp.array(jax.tree_util.tree_flatten(state['act_size'])[0])
      ) / size_div
  # metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int):
  """Create learning rate schedule."""
  if config.lr_boundaries_scale is not None:
    schedule_fn = optax.piecewise_constant_schedule(config.learning_rate, {int(
        k) * steps_per_epoch: v for k, v in config.lr_boundaries_scale.items()}
    )
  else:
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=cosine_epochs * steps_per_epoch)

    if config.warmup_epochs != 0.:
      warmup_fn = optax.linear_schedule(
          init_value=0., end_value=base_learning_rate,
          transition_steps=config.warmup_epochs * steps_per_epoch)
      schedule_fn = optax.join_schedules(
          schedules=[warmup_fn, cosine_fn],
          boundaries=[config.warmup_epochs * steps_per_epoch])
    else:
      schedule_fn = cosine_fn
  return schedule_fn


def parametric_d_xmax_is_leaf(x):
  if type(x) is dict or type(x) is flax.core.frozen_dict.FrozenDict:
    if 'dynamic_range' in x:
      return True
    return False
  return True


def clip_single_leaf_params(x, quant_config):

  if type(x) is dict or type(x) is flax.core.frozen_dict.FrozenDict:
    if 'dynamic_range' in x and 'step_size' in x:

      min_value = jnp.minimum(x['step_size'], x['dynamic_range'] - 1e-5)
      max_value = jnp.maximum(x['step_size'] + 1e-5, x['dynamic_range'])
      x['step_size'] = min_value
      x['dynamic_range'] = max_value

      x['step_size'] = jnp.clip(
          x['step_size'], quant_config['min_d'] + 1e-5,
          quant_config['max_d'] - 1e-5)
      x['dynamic_range'] = jnp.clip(
          x['dynamic_range'], quant_config['min_xmax'] + 1e-5,
          quant_config['max_xmax'] - 1e-5)

  return x


def clip_quant_vals(params, quant_configs):
  if len(quant_configs.keys()) == 0:
    return params
  quant_configs = unfreeze(quant_configs)
  quant_configs['placeholder'] = jnp.sum(jnp.ones((1,)))
  quant_configs = freeze(quant_configs)
  return jax.tree_util.tree_multimap(clip_single_leaf_params, params,
                                     quant_configs,
                                     is_leaf=parametric_d_xmax_is_leaf)


def clip_single_leaf_grads(x, params):
  if type(x) is dict or type(x) is flax.core.frozen_dict.FrozenDict:
    if 'dynamic_range' in x and 'step_size' in x:
      x['dynamic_range'] = jnp.clip(
          x['dynamic_range'], -params['step_size'], +params['step_size'])
      x['step_size'] = jnp.clip(
          x['step_size'], -params['step_size'], +params['step_size'])

    for key in x.keys():
      if 'no_train' in key:
        print(key)
        x[key] = jnp.zeros_like(x[key])
  return x


def clip_quant_grads(grads, quant_params):
  return jax.tree_util.tree_multimap(clip_single_leaf_grads, grads,
                                     quant_params,
                                     is_leaf=parametric_d_xmax_is_leaf)


def weight_decay_fn(params):
  l2_params = [p for ((mod_name), p) in tree.flatten_with_path(
      params) if 'BatchNorm' not in str(mod_name) and 'bn_init'
      not in str(mod_name) and 'stem_bn' not in str(mod_name)
      and 'head_bn' not in str(mod_name)]
  return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


def train_step(state, batch, rng, learning_rate_fn, weight_decay,
               quant_target, smoothing, no_quant_params=False):
  """Perform a single training step."""
  rng, prng = jax.random.split(rng, 2)

  def loss_fn(params, inputs, targets, quant_params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn({'params': params,
                                              'quant_params': quant_params,
                                              'batch_stats': state.batch_stats,
                                              'weight_size': state.weight_size,
                                              'act_size': state.act_size,
                                              'quant_config':
                                              state.quant_config},
                                             inputs, rng=prng, mutable=[
        'batch_stats', 'weight_size', 'act_size', 'intermediates',
        'quant_config'],
        rngs={'dropout': rng})

    loss = jnp.mean(cross_entropy_loss(logits, targets, smoothing))
    loss += weight_decay * weight_decay_fn(params)

    # size penalty
    size_weight_penalty = 0.
    size_act_penalty = 0.
    if hasattr(quant_target, 'weight_mb'):
      size_weight = jnp.sum(jnp.array(jax.tree_util.tree_flatten(
          new_model_state['weight_size'])[0])) / quant_target.size_div
      size_weight_penalty += quant_target.weight_penalty * jax.nn.relu(
          size_weight - quant_target.weight_mb) ** 2
    if hasattr(quant_target, 'act_mb'):
      if quant_target.act_mode == 'sum':
        size_act = jnp.sum(jnp.array(jax.tree_util.tree_flatten(
            new_model_state['act_size'])[0])) / quant_target.size_div
        size_act_penalty += quant_target.act_penalty * jax.nn.relu(
            size_act - quant_target.act_mb) ** 2
      elif quant_target.act_mode == 'max':
        size_act = max_custom_grad(jnp.array(jax.tree_util.tree_flatten(
            new_model_state['act_size'])[0])) / quant_target.size_div
        size_act_penalty += quant_target.act_penalty * \
            jax.nn.relu(size_act - quant_target.act_mb) ** 2
      else:
        raise Exception(
            'Unrecongized quant act mode, either sum or \
            max but got: ' + quant_target.act_mode)

    final_loss = loss + size_act_penalty + size_weight_penalty
    return final_loss, (new_model_state, logits, final_loss,
                        size_act_penalty, size_weight_penalty, loss)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, argnums=[0, 3], has_aux=True)
  aux, grads = grad_fn(
      state.params['params'], batch['image'], batch['label'],
      state.params['quant_params'])

  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = (grads[0], clip_quant_grads(grads[1], state.params['quant_params']))
  grads = lax.pmean(grads, axis_name='batch')

  if no_quant_params:
    grads = (grads[0], jax.tree_util.tree_map(
        lambda x: jnp.zeros_like(x), grads[1]))

  new_model_state, logits, _, _, _, _ = aux[1]

  metrics = compute_metrics(
      logits, batch['label'], new_model_state, quant_target.size_div,
      smoothing)
  metrics['learning_rate'] = lr
  new_state = state.apply_gradients(
      grads={'params': grads[0], 'quant_params': grads[1]},
      batch_stats=new_model_state['batch_stats'],
      weight_size=new_model_state['weight_size'],
      act_size=new_model_state['act_size'],
      quant_config=new_model_state['quant_config'])

  new_state.params['quant_params'] = clip_quant_vals(
      new_state.params['quant_params'], new_state.quant_config)
  metrics['final_loss'] = aux[1][-4]
  metrics['size_act_penalty'] = aux[1][-3]
  metrics['size_weight_penalty'] = aux[1][-2]
  metrics['ce_loss'] = aux[1][-1]
  metrics['accuracy'] = metrics['accuracy'].mean()

  return new_state, metrics


def eval_step(state, batch, size_div, smoothing):
  variables = {'params': state.params['params'],
               'quant_params': state.params['quant_params'],
               'batch_stats': state.batch_stats,
               'weight_size': state.weight_size,
               'act_size': state.act_size, 'quant_config': state.quant_config}
  logits, new_state = state.apply_fn(
      variables,
      batch['image'],
      rng=jax.random.PRNGKey(0),
      train=False,
      mutable=['weight_size', 'act_size'])
  metrics = compute_metrics(
      logits, batch['label'], new_state, size_div, smoothing)
  metrics['accuracy'] = metrics['accuracy'].mean()
  return metrics


class TrainState(train_state.TrainState):
  batch_stats: Any
  weight_size: Any
  act_size: Any
  quant_config: Any


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3, overwrite=True)


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

  (
      params,
      quant_params,
      batch_stats,
      weight_size,
      act_size,
      quant_config
  ) = initialized(rng, image_size, model)
  if config.optimizer == 'rmsprop':
    tx = optax.rmsprop(
        learning_rate=learning_rate_fn,
        decay=0.9,
        momentum=config.momentum,
        eps=0.001,
    )
  elif config.optimizer == 'sgd':
    tx = optax.sgd(
        learning_rate=learning_rate_fn,
        momentum=config.momentum,
        nesterov=config.nesterov,
    )
  elif config.optimizer == 'adam':
    tx = optax.adam(
        learning_rate=learning_rate_fn,
    )
  else:
    raise Exception('Unknown optimizer in config: ' + config.optimizer)
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
      quant_config=quant_config,
  )
  return state
