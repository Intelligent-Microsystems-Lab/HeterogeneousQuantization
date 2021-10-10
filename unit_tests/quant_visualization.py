from quant import parametric_d
import sys

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

sys.path.append("..")


quant_fn = parametric_d(bits=2)

rng = jax.random.PRNGKey(0)
rng, init_rng, data_rng = jax.random.split(rng, 3)

params = quant_fn.init(init_rng, jnp.ones((1, 2))*.25, sign=True)


def loss_fn(x, params):
  logits = quant_fn.apply(params, x, sign=True)
  return jnp.sum(logits)


grad_fn = jax.grad(loss_fn, argnums=1)

gs_list = []
x_list = jnp.arange(-1, +1, .001)
for i in x_list:
  g = grad_fn(i, params)
  gs_list.append(g['quant_params']['step_size'])

grad_fn = jax.grad(loss_fn, argnums=0)

gx_list = []
for i in x_list:
  g = grad_fn(i, params)
  gx_list.append(g)


font_size = 26

fig, ax = plt.subplots(figsize=(10, 6.8))
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

ax.plot(x_list, gs_list,
        color='blue', label='Derivative Step Size', linewidth=5)
ax.plot(x_list, gx_list,
        color='green', label='Derivative Inputs', linewidth=5)


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
plt.savefig("../../figures/lsq_gradients.png")
plt.close()
