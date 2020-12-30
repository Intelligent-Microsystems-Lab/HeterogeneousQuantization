import argparse, time, pickle, uuid

from functools import partial
import jax.numpy as jnp
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax

import matplotlib.pyplot as plt

# module load python cuda/10.2
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"

@custom_vjp
def spike_nonlinearity(u, thr):
    return  (u > thr).astype(jnp.float32)

def spike_nonlinearity_fwd(u, thr):
    return  (u > thr).astype(jnp.float32), (u, thr)

def spike_nonlinearity_bwd(ctx, g):
    u, thr = ctx
    return (g * vmap(grad(jax.nn.sigmoid))(u - thr),None,)
    #return (g/(10*jnp.abs(u)+1.)**2,)

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
    c_pred = convt_run(alpha_vr, pred, jnp.zeros(so_size))
    c_target = convt_run(alpha_vr, target, jnp.zeros(so_size))

    return jnp.reshape(jnp.sqrt(1/5e-3*jnp.sum((c_pred - c_target)**2)), ())

def pattern_plot(sin, sout, pred, name):
    fig, axes = plt.subplots(nrows=1, ncols=4)
    axes[0].imshow(sin.astype(float))
    axes[0].set_title("Input")
    axes[2].imshow(pred.astype(float))
    axes[2].set_title("Output")
    axes[1].imshow(sout.astype(float))
    axes[1].set_title("Target")
    axes[3].imshow(pred.astype(float) - sout.astype(float))
    axes[3].set_title("Error")

    plt.tight_layout()
    plt.savefig('figures/'+name+'.png')
    plt.close()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--log-file", type=str, default="logs/test.csv", help='Log-file')
parser.add_argument("--seed", type=int, default=80085, help='Random seed')

parser.add_argument("--input", type=str, default="data/smile_data_set/input_700_250_25.pkl", help='Input pickle')
parser.add_argument("--target", type=str, default="data/smile_data_set/smile95.pkl", help='Target pattern pickle')
parser.add_argument("--architecture", type=str, default="700-400-250", help='Architecture of the networks')

parser.add_argument("--l_rate", type=float, default=1e-6, help='Learning Rate')
parser.add_argument("--epochs", type=int, default=10000, help='Epochs')

parser.add_argument("--alpha", type=float, default=.96, help='Time constant for membrane potential')
parser.add_argument("--alpha_vr", type=float, default=.75, help='Time constant for Van Rossum distance')
parser.add_argument("--thr", type=float, default=.1, help='Membrane Threshold')

args = parser.parse_args()


key = random.PRNGKey(args.seed)
model_uuid = str(uuid.uuid4())


with open(args.input, 'rb') as f:
    x_train = jnp.array(pickle.load(f)).transpose()
with open(args.target, 'rb') as f:
    y_train = jnp.array(pickle.load(f))


layers = jnp.array(list(map(int, args.architecture.split("-"))))
n_layers = len(layers)

weights = []
biases = []
for i in range(len(layers)-1):
    key, subkey1 = random.split(key, 2)
    weights.append(random.normal(subkey1, (layers[i+1], layers[i])) * jnp.sqrt(6/(layers[i] + layers[i+1])))
    biases.append(jnp.zeros(layers[i+1]))


def run_snn(weights, biases, alpha, thr, x_train):
    T = x_train.shape[0]
    mem = [jnp.zeros(l.shape[0]) for l in weights]
    out_s = jnp.empty((0, weights[-1].shape[0]))

    for t in range(T):
        st = x_train[t,:]
        for i, (w, b) in enumerate(zip(weights, biases)):
            mem[i] = alpha * mem[i] + jnp.dot(weights[i], st) + biases[i]
            st = spike_nonlinearity(mem[i], thr)
            mem[i] -= st
        out_s = jnp.vstack((out_s, st))
    return out_s

@jax.partial(jit, static_argnums=[2, 3, 4])
def loss_pred(weights, biases, alpha, alpha_vr, thr, x_train, y_train):
    pred_s = run_snn(weights, biases, alpha, thr, x_train)
    return vr_loss(alpha_vr, pred_s, y_train)

print(model_uuid)
print(args)
loss_hist = []
for e in range(args.epochs):
    loss, gwb = value_and_grad(loss_pred, argnums= (0,1))(weights, biases, args.alpha, args.alpha_vr, args.thr, x_train, y_train)
    weights = [w - args.l_rate * gw.sum(0) for w, gw in zip(weights, gwb[0])]
    biases = [b - args.l_rate * gb.sum(0) for b, gb in zip(biases, gwb[1])]
    loss_hist.append(loss)
    print("{} {:.4f}".format(e, loss))

# Visualization
#pred = run_snn(weights, biases, args.alpha, args.thr, x_train)
#pattern_plot(x_train, y_train, pred, model_uuid + "_pattern_visual")

# save model
jnp.savez("models/"+str(model_uuid)+".npz", weights, biases, loss_hist, args)
# add log entry
with open(args.log_file,'a') as f:
    f.write(str(loss_hist[-1]) + "," + model_uuid + "," + ",".join( [str(vars(args)[t]) for t in vars(args)]) + "\n")
