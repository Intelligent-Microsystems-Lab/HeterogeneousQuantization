import argparse, time, pickle, uuid, os, datetime

from functools import partial
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax


@custom_vjp
def spike_nonlinearity(u, thr):
    return  (u > thr).astype(jnp.float32)

def spike_nonlinearity_fwd(u, thr):
    return  (u > thr).astype(jnp.float32), (u, thr)

def spike_nonlinearity_bwd(ctx, g):
    u, thr = ctx
    return (g * vmap(grad(jax.nn.sigmoid))(u - thr),None,)
    #return (g/(10*jnp.abs(u)+1.)**2,None,)

spike_nonlinearity.defvjp(spike_nonlinearity_fwd, spike_nonlinearity_bwd)


# Zenke trick, ignore reset in bptt - untested
@custom_vjp
def reset_mem(u, s, thr):
    return u - s * thr

def reset_mem_fwd(u, thr):
    return  u - s * thr, None

def reset_mem_bwd(ctx, g):
    return (g, None, None,)

reset_mem.defvjp(reset_mem_fwd, reset_mem_bwd)

def convt(alpha_vr, state, signal):
    return signal + (alpha_vr * state)

def convt_scan(alpha_vr, state, signal):
    h = convt(alpha_vr, state, signal)
    return h, h

def convt_run(alpha_vr, signal, s0):
    s = s0
    f = partial(convt_scan, alpha_vr)
    _, h_t = lax.scan(f, s, signal)
    return h_t

def vr_loss(alpha_vr, pred, target):
    so_size = target.shape[1]
    c_pred = vmap(convt_run, (None, 0, None))(alpha_vr, pred, jnp.zeros(so_size))
    c_target = vmap(convt_run, (None, 0, None))(alpha_vr, target, jnp.zeros(so_size))
    
    return jnp.sqrt(1/5e-3*jnp.sum((c_pred - c_target)**2))

def nll_loss(alpha_vr, pred, target):
    one_hot = target.mean(1)
    prob = pred.mean(1)

    logits = prob - jax.scipy.special.logsumexp(prob, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(logits * one_hot, axis=1))

def smooth_l1(alpha_vr, pred, target):
    pass

def one_step(weights, biases, alpha, thr, gamma, mem, st):
    for i, (w, b) in enumerate(zip(weights, biases)):
        mem[i] = alpha * mem[i] + jnp.dot(weights[i], st) + biases[i]
        st = spike_nonlinearity(mem[i], thr)
        mem[i] -= st * gamma
    return mem, st

def run_snn(weights, biases, alpha, gamma, thr, x_train):
    mem = [jnp.zeros(l.shape[0]) for l in weights]

    f = partial(one_step, weights, biases, alpha, thr, gamma)
    mem, out_s = lax.scan(f, mem, x_train)

    return out_s

v_run_snn = jit(vmap(run_snn, (None, None, None, None, None, 0)), static_argnums=[2, 3, 4])

def loss_pred(weights, biases, alpha, gamma, alpha_vr, thr, x_train, y_train, loss_fn):
    pred_s = v_run_snn(weights, biases, alpha, gamma, thr, x_train)
    return loss_fn(alpha_vr, pred_s, y_train)

@jax.partial(jit, static_argnums=[1, 2, 3, 4, 5, 6, 9])
def update_w(opt_state, get_params, opt_update, alpha, gamma, alpha_vr, thr, x_train, y_train, loss_fn, e):
    loss, gwb = value_and_grad(loss_pred, argnums= (0,1))(get_params(opt_state)[0], get_params(opt_state)[1], alpha, gamma, alpha_vr, thr, x_train, y_train, loss_fn)
    opt_state = opt_update(e, gwb, opt_state)
    
    return loss, opt_state, get_params(opt_state)[0], get_params(opt_state)[1]

@jit
def acc_compute(pred, target):
    return jnp.mean(pred.sum(1).argmax(1) == target.sum(1).argmax(1))
