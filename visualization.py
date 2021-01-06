import argparse, time, pickle, uuid, os, datetime

from functools import partial
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax

import matplotlib.pyplot as plt



def pattern_plot(sin, sout, pred, name, textl):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=4)
    axes[0].imshow(sin.astype(float))
    axes[0].set_title("Input")
    axes[2].imshow(pred.astype(float))
    axes[2].set_title("Output")
    axes[1].imshow(sout.astype(float))
    axes[1].set_title("Target")
    axes[3].imshow(jnp.abs(pred.astype(float) - sout.astype(float)))
    axes[3].set_title("Error")

    plt.suptitle(textl)
    plt.tight_layout()
    plt.savefig('figures/'+name+'.png')
    plt.close()


def yy_plot(sin, pred, name, textl):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=1)
    x_points = sin.argmax(1)/sin.shape[1]
    cla_col = ['orange','blue','green']
    for i in range(3):
        plt.scatter(x_points[pred.sum(1).argmax(1) == i][:,0], x_points[pred.sum(1).argmax(1) == i][:,1], color = cla_col[i]  )
        plt.scatter(1-x_points[pred.sum(1).argmax(1) == i][:,2], 1-x_points[pred.sum(1).argmax(1) == i][:,3], color = cla_col[i]  )

    plt.title(textl)
    plt.tight_layout()
    plt.savefig('figures/'+name+'.png')
    plt.close()

def curve_plot(loss_hist, train_hist, test_hist, name, textl):
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=1)

    axes.plot(train_hist, label='Train', color='blue')
    axes.plot(test_hist, label = 'Test', color = 'red')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    ax2 = axes.twinx()  

    ax2.plot(loss_hist, label = 'Loss', color='black')
    #ax2.ylabel("Loss")

    plt.suptitle(textl)
    plt.tight_layout()
    plt.savefig('figures/'+name+'.png')
    plt.close()