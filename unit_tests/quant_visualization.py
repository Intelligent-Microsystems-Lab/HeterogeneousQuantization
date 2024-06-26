# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Visualitzation for quantization.

import jax
import jax.numpy as jnp
import functools

import matplotlib.pyplot as plt

import sys
sys.path.append("..")

from quant import (  # noqa: E402
    uniform_static,
    parametric_d_xmax,

    round_ewgs,
    round_acos,
    round_tanh,
    round_invtanh,
    round_psgd,
)


def passthrough(x, scale, off=False):
  return round_ewgs(x, scale=0.)


def plot_lines(x, x_list, name_list, color_list, fname):
  font_size = 22

  fig, ax = plt.subplots(figsize=(12, 8.8))
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  ax.xaxis.set_tick_params(width=5, length=10, labelsize=font_size)
  ax.yaxis.set_tick_params(width=5, length=10, labelsize=font_size)

  for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(5)

  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')

  for x_val, name, color in zip(x_list, name_list, color_list):
    ax.plot(x, x_val,
            color=color, label=name, linewidth=5)

  ax.set_xlabel("x", fontsize=font_size, fontweight='bold')
  ax.set_ylabel("dL/dx", fontsize=font_size, fontweight='bold')
  plt.legend(
      bbox_to_anchor=(0.5, 1.2),
      loc="upper center",
      ncol=2,
      frameon=False,
      prop={'weight': 'bold', 'size': font_size}
  )
  plt.tight_layout()
  plt.savefig(fname)
  plt.close()


# Ordinary visualization

def vis_surrogate(sur_fn):
  quant_fn = uniform_static(
      bits=2, round_fn=sur_fn, g_scale=1)

  rng = jax.random.PRNGKey(0)
  rng, init_rng, data_rng = jax.random.split(rng, 3)

  params = quant_fn.init(init_rng, jax.random.uniform(
      data_rng, (100, 100)), sign=True)

  def loss_fn(x, params):
    logits = quant_fn.apply(params, x, sign=True)
    return jnp.sum(logits)

  x_list = jnp.arange(-1.1, +1.1, .00001)
  grad_fn = jax.grad(loss_fn, argnums=0)

  g = grad_fn(x_list, params)
  plot_lines(x_list, [g], ['Derivative Inputs'], ['blue', ],
             "../../figures/quant_grad_" + sur_fn.__name__ + ".png")


# vis_surrogate(passthrough)
# vis_surrogate(round_ewgs)
vis_surrogate(round_tanh)
vis_surrogate(round_acos)
# vis_surrogate(round_invtanh)
# vis_surrogate(round_psgd)


# MixedPrecision DNN visualization -- Surrogate
def vis_surrogate_mixed(sur_fn):
  quant_fn = parametric_d_xmax(bits=3, round_fn=sur_fn, g_scale=1.)

  rng = jax.random.PRNGKey(0)
  rng, init_rng, data_rng = jax.random.split(rng, 3)

  params = quant_fn.init(init_rng, jnp.ones((1, 2)), sign=True)

  def loss_fn(x, params):
    logits = quant_fn.apply(params, x, sign=True)
    return jnp.sum(logits)

  grad_fn = jax.grad(loss_fn, argnums=1)

  gs_list = []
  gxmax_list = []
  x_list = jnp.arange(-1.1, +1.1, .001)
  for i in x_list:
    g = grad_fn(i, params)
    gs_list.append(g['quant_params']['step_size'])
    gxmax_list.append(g['quant_params']['dynamic_range'])

  grad_fn = jax.grad(loss_fn, argnums=0)

  gx_list = []
  for i in x_list:
    g = grad_fn(i, params)
    gx_list.append(g)

  plot_lines(x_list, [gx_list, gxmax_list, gs_list],
             ['Derivative Inputs', 'Derivative Dynamic Range',
              'Derivative Step Size'], ['blue', 'green', 'red'],
             "../../figures/mixed_surrogate_" + sur_fn.__name__ + ".png")


# vis_surrogate_mixed(passthrough)
# vis_surrogate_mixed(round_ewgs)
# vis_surrogate_mixed(round_tanh)
# vis_surrogate_mixed(round_invtanh)
