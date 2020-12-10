import argparse, time, pickle
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
from jax.experimental import optimizers, stax

import matplotlib.pyplot as plt



def pattern_plot(inp, out, pred):
    fig, axes = plt.subplots(nrows=1, ncols=4)
    axes[0].imshow(inp.astype(float))
    axes[0].set_title("Input")
    axes[2].imshow(out.astype(float))
    axes[2].set_title("Output")
    axes[1].imshow(pred.astype(float))
    axes[1].set_title("Target")
    axes[3].imshow(pred.astype(float) - out.astype(float))
    axes[3].set_title("Error")

    plt.tight_layout()
    plt.savefig('figures/pattern_visualization.png')
    plt.close()