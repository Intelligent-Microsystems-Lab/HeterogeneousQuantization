import argparse, time, pickle
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
from jax.experimental import optimizers, stax


# quant functions
def step_d(bits): 
    return 2.0 ** (bits - 1)

def shift(x):
    if x == 0:
        return 1
    return 2 ** jnp.round(jnp.log2(x))

def clip(x, bits):
    delta = 1./step_d(bits)
    maxv = +1 - delta
    minv = -1 + delta
    return jnp.clip(x, minv, maxv)

def quant(x, bits):
    scale = step_d(bits)
    return jnp.round(x * scale ) / scale

def cq(x, bits):
    return quant(clip(x, bits), bits)

@custom_vjp
def quant_pass(u, bw):
    if bw[0] == 'fp':
        return u
    return quant(u, bw[0])

def quant_pass_fwd(u, bw):
    return  quant_pass(u, bw), bw[1]

def quant_pass_bwd(eb, g):
    if eb == 'fp':
        return (g, None)
    alpha = shift(jnp.max(jnp.abs(g)))
    return (cq(g/alpha, eb), None)

quant_pass.defvjp(quant_pass_fwd, quant_pass_bwd)


# @custom_vjp
# def quant_mvm(u, bw):


#     return 

# def quant_mvm_fwd(u, bw):
#     return  quant_mvm(u, bw), bw[1]

# def quant_mvm_bwd(eb, g):
# 	alpha = shift(jnp.max(jnp.abs(g)))
#     return (cq(g/alpha, eb) , None)

# quant_mvm.defvjp(quant_mvm_fwd, quant_mvm_bwd)


@custom_vjp
def spike_nonlinearity(u):
    return  (u > 0).astype(jnp.float32)

def spike_nonlinearity_fwd(u):
    return  (u > 0).astype(jnp.float32), u

def spike_nonlinearity_bwd(u, g):
    return (g/(10*jnp.abs(u)+1.)**2,)

spike_nonlinearity.defvjp(spike_nonlinearity_fwd, spike_nonlinearity_bwd)


def convt(params_fixed, state, signal):
    return signal + (params_fixed['alpha_vr'] * state)

def convt_scan(params_fixed, state, signal):
    h = convt(params_fixed, state, signal)
    return h, h

def convt_run(params_fixed, signal, s0):
    s = s0
    f = partial(convt_scan, params_fixed)
    _, h_t = lax.scan(f, s, signal)
    return h_t

def snn_c_fwd(params_fixed, params, u, s):
    # single SNN layer - resting potential 0
    mem = jnp.dot(params['w'], u[1]) - params_fixed['delta'] * u[2]
    s_out = spike_nonlinearity(mem - params_fixed['thr'])
    q, p, r  = params_fixed['beta'] * u[0] + s, params_fixed['alpha'] * u[1] + u[0], params_fixed['gamma'] * u[2] + s_out

    return (q, p, r), s_out


def ssn_fc_net(params_fixed, params, ut, st):

    # 1st FC SNN layer
    u1, x = snn_c_fwd(params_fixed, params[0], ut[0], st)

    # 2nd FC SNN layer
    u2, x = snn_c_fwd(params_fixed, params[1], ut[1], x)

    # 2nd FC SNN layer
    u3, x = snn_c_fwd(params_fixed, params[2], ut[2], x)

    return [u1, u2, u3], x 

def state_init(params):
    u = []
    for i in params:
        u.append(( jnp.zeros(i['w'].shape[1]),  jnp.zeros(i['w'].shape[1]),  jnp.zeros(i['w'].shape[0])))
    return u

def snn_fc_net_run(params_fixed, params, s_in, u0):
    u = u0
    f = partial(ssn_fc_net, params_fixed, params)
    _, s_out = lax.scan(f, u, s_in)
    return s_out


def vr_loss(params_fixed, params, s_in, target):
    so_size = target.shape[1]
    state0 = state_init(params)

    pred = snn_fc_net_run(params_fixed, params, s_in, state0)

    c_pred = convt_run(params_fixed, pred, jnp.zeros(so_size))
    c_target = convt_run(params_fixed, target, jnp.zeros(so_size))

    return jnp.reshape(jnp.sqrt(1/5e-3*jnp.sum((c_pred - c_target)**2)), ())