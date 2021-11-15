# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Originally copied from https://github.com/google/flax/ and minimally changed

from typing import (Any, Callable, Optional, Tuple)

import jax
from jax import lax
from jax.nn import initializers
import jax.numpy as jnp

from flax.linen.module import Module, compact, merge_param


PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?


def _no_init(rng, shape):
  return ()


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(Module):
  """BatchNorm Module.

  Usage Note:
  If we define a model with BatchNorm, for example::

    BN = nn.BatchNorm(use_running_average=False, momentum=0.9, epsilon=1e-5,
                      dtype=jnp.float32)

  The initialized variables dict will contain in addition to a 'params'
  collection a separate 'batch_stats' collection that will contain all the
  running statistics for all the BatchNorm layers in a model::

    vars_initialized = BN.init(key, x)  # {'params': ..., 'batch_stats': ...}

  We then update the batch_stats during training by specifying that the
  `batch_stats` collection is mutable in the `apply` method for our module.::

    vars_in = {'params': params, 'batch_stats': old_batch_stats}
    y, mutated_vars = BN.apply(vars_in, x, mutable=['batch_stats'])
    new_batch_stats = mutated_vars['batch_stats']

  During eval we would define BN with `use_running_average=True` and use the
  batch_stats collection from training to set the statistics.  In this case
  we are not mutating the batch statistics collection, and needn't mark it
  mutable::

    vars_in = {'params': params, 'batch_stats': training_batch_stats}
    y = BN.apply(vars_in, x)

  Attributes:
    use_running_average: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For
      example, `[[0, 1], [2, 3]]` would independently batch-normalize over
      the examples on the first two and last two devices. See `jax.lax.psum`
      for more details.
  """
  use_running_average: Optional[bool] = None
  axis: int = -1
  momentum: float = 0.99
  epsilon: float = 1e-5
  dtype: Dtype = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
  scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
  axis_name: Optional[str] = None
  axis_index_groups: Any = None

  @compact
  def __call__(self, x, use_running_average: Optional[bool] = None):
    """Normalizes the input using batch statistics.

    NOTE:
    During initialization (when parameters are mutable) the running average
    of the batch statistics will not be updated. Therefore, the inputs
    fed during initialization don't need to match that of the actual input
    distribution and the reduction axis (set with `axis_name`) does not have
    to exist.

    Args:
      x: the input to be normalized.
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.

    Returns:
      Normalized inputs (the same shape as inputs).
    """
    use_running_average = merge_param(
        'use_running_average', self.use_running_average, use_running_average)
    x = jnp.asarray(x, jnp.float32)
    axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
    axis = _absolute_dims(x.ndim, axis)
    feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
    reduced_feature_shape = tuple(
        d for i, d in enumerate(x.shape) if i in axis)
    reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

    # see NOTE above on initialization behavior
    initializing = self.is_mutable_collection('params')

    ra_mean = self.variable('batch_stats', 'mean',
                            lambda s: jnp.zeros(s, jnp.float32),
                            reduced_feature_shape)
    ra_var = self.variable('batch_stats', 'var',
                           lambda s: jnp.ones(s, jnp.float32),
                           reduced_feature_shape)

    if use_running_average:
      mean, var = ra_mean.value, jax.nn.relu(ra_var.value)
    else:
      mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
      mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
      if self.axis_name is not None and not initializing:
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=self.axis_name,
                axis_index_groups=self.axis_index_groups), 2)
      var = jax.nn.relu(mean2 - lax.square(mean))

      if not initializing:
        ra_mean.value = self.momentum * \
            ra_mean.value + (1 - self.momentum) * mean
        ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

    y = x - mean.reshape(feature_shape)
    mul = lax.rsqrt(var + self.epsilon).reshape(feature_shape)
    if self.use_scale:
      scale = self.param('scale',
                         self.scale_init,
                         reduced_feature_shape).reshape(feature_shape)
      mul = mul * scale
    y = y * mul
    if self.use_bias:
      bias = self.param('bias',
                        self.bias_init,
                        reduced_feature_shape).reshape(feature_shape)
      y = y + bias
    return jnp.asarray(y, self.dtype)
