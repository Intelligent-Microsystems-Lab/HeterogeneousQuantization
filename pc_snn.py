import argparse, time, pickle, uuid, os, datetime

from functools import partial
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax

import matplotlib.pyplot as plt

from datasets import dl_create
from visualization import pattern_plot, yy_plot, curve_plot
from snn_util import v_run_snn, update_w, acc_compute

# module load python cuda/10.2
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"


@custom_vjp
def spike_nonlinearity(u, thr = 1.):
    return  (u >= thr).astype(jnp.float32)

def spike_nonlinearity_fwd(u, thr = 1.):
    return  (u >= thr).astype(jnp.float32), (u, thr)

def spike_nonlinearity_bwd(ctx, g):
    u, thr = ctx
    return (g * grad(jax.nn.sigmoid)(u - thr),None,)
    #return (g/(10*jnp.abs(u)+1.)**2,None,)

spike_nonlinearity.defvjp(spike_nonlinearity_fwd, spike_nonlinearity_bwd)


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


def infer_one(alpha, thr, gamma, weights, biases, loop_carry, sin, sout):
    x, f_n, f_p, mem, error, grad_w, grad_b = loop_carry
    n_layers = len(weights) + 1 

    # set new input and target
    x[0] = sin 
    x[-1] = sout * thr
    # calculate errors
    for i in range(1,n_layers):
        f_n[i-1] = act_fn(x[i-1])
        f_p[i-1] = vmap(grad(act_fn))(x[i-1])
        mem[i]   = (alpha * mem[i]) + (jnp.dot(weights[i-1], f_n[i-1]) - biases[i-1]) - (gamma * act_fn(x[i]))
        error[i] = (x[i] - mem[i])
    # update variable nodes
    for l in range(1,n_layers-1):
        g = jnp.dot(weights[l].transpose(), error[l+1]) * f_p[l]
        x[l] = x[l] + beta * (-error[l] + g)

    # calculate gradients
    for i in range(n_layers-1):
        grad_b[i] += -error[i+1]
        grad_w[i] += -jnp.outer(error[i+1], act_fn(x[i]).transpose())


    return (x, f_n, f_p, mem, error, grad_w, grad_b), (grad_w, grad_b)

def infer_pc(sin, sout, weights, biases, beta, alpha, gamma, thr, loss_fn): #var_layer
    n_layers = len(weights) + 1 
    act_fn = partial(spike_nonlinearity, thr = thr)

    error = [[] for i in range(n_layers)]
    f_n = [[] for i in range(n_layers)]
    f_p = [[] for i in range(n_layers)]
    mem = [[] for i in range(n_layers)]
    x =  [[] for i in range(n_layers)]

    grad_w = [jnp.zeros_like(i) for i in weights]
    grad_b = [jnp.zeros_like(i) for i in biases]

    for i in range(n_layers-1):
        mem[i+1] = jnp.zeros(weights[i].shape[0])
        x[i+1] = jnp.zeros(weights[i].shape[0])

    # infer function
    for t in range(sin.shape[0]):
        # set new input and target
        x[0] = sin[t,:] 
        x[-1] = sout[t,:] * thr
        # calculate errors
        for i in range(1,n_layers):
            f_n[i-1] = act_fn(x[i-1])
            f_p[i-1] = vmap(grad(act_fn))(x[i-1])
            mem[i]   = (alpha * mem[i]) + (jnp.dot(weights[i-1], f_n[i-1]) - biases[i-1]) - (gamma * act_fn(x[i]))
            error[i] = (x[i] - mem[i])
        # update variable nodes
        for l in range(1,n_layers-1):
            g = jnp.dot(weights[l].transpose(), error[l+1]) * f_p[l]
            x[l] = x[l] + beta * (-error[l] + g)

        # calculate gradients
        for i in range(n_layers-1):
            grad_b[i] += -error[i+1]
            grad_w[i] += -jnp.outer(error[i+1], act_fn(x[i]).transpose())

    grad_w = [w/sin.shape[0] for w in grad_w]
    grad_b = [b/sin.shape[0] for b in grad_b]

    return x, error, grad_w, grad_b

v_infer_pc = vmap(infer_pc, (0, 0, None, None, None, None, None, None, None))

def learn_pc(sin, sout, weights, biases, alpha, thr, gamma, beta, loss_fn):
    x, error, grad_w, grad_b = infer_pc(sin, sout, weights, biases, beta, alpha, gamma, thr, loss_fn)

    return jnp.abs(error[-1].sum()), (grad_w, grad_b)

# vmap learn_pc

#@jax.partial(jit, static_argnums=[1, 2, 3, 4, 5, 6, 9])
def update_w(opt_state, get_params, opt_update, alpha, gamma, beta, thr, x_train, y_train, loss_fn, e):
    loss, gwb = learn_pc(x_train[0,:,:], y_train[0,:,:], weights, biases, alpha, thr, gamma, beta, loss_fn)
    opt_state = opt_update(e, gwb, opt_state)
    
    return loss, opt_state, get_params(opt_state)[0], get_params(opt_state)[1]

@jit
def acc_compute(pred, target):
    return jnp.mean(pred.sum(1).argmax(1) == target.sum(1).argmax(1))


def nll_loss(alpha_vr, pred, target):
    one_hot = target.mean(1)
    prob = pred.mean(1)

    logits = prob - jax.scipy.special.logsumexp(prob, axis=1, keepdims=True)
    return -jnp.mean(jnp.sum(logits * one_hot, axis=1))


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--log-file", type=str, default="logs/test.csv", help='Log-file')
parser.add_argument("--seed", type=int, default=80085, help='Random seed')

# parser.add_argument("--data-set", type=str, default="Smile", help='Data set to use')
# parser.add_argument("--architecture", type=str, default="700-500-250", help='Architecture of the networks')

# parser.add_argument("--data-set", type=str, default="Yin_Yang", help='Data set to use')
# parser.add_argument("--architecture", type=str, default="4-120-3", help='Architecture of the networks')

parser.add_argument("--data-set", type=str, default="NMNIST", help='Data set to use')
parser.add_argument("--architecture", type=str, default="2048-500-10", help='Architecture of the networks')

# parser.add_argument("--data-set", type=str, default="DVS_Gestures", help='Data set to use')
# parser.add_argument("--architecture", type=str, default="2048-500-11", help='Architecture of the networks')

parser.add_argument("--l_rate", type=float, default=1e-3, help='Learning Rate')
parser.add_argument("--epochs", type=int, default=10000, help='Epochs')

parser.add_argument("--w-scale", type=float, default=2., help='Weight Scaling')
parser.add_argument("--batch-size", type=float, default=128, help='Batch Size ')

parser.add_argument("--alpha", type=float, default=.95, help='Time constant for membrane potential')
parser.add_argument("--gamma", type=float, default=1.2, help='Reset Magnitude')
parser.add_argument("--thr", type=float, default=1., help='Membrane Threshold')
parser.add_argument("--beta", type=float, default=.2, help='Euler integration constant')

args = parser.parse_args()


key = random.PRNGKey(args.seed)
model_uuid = str(uuid.uuid4())

train_dl, test_dl, loss_fn = dl_create(args.data_set, args.batch_size)

layers = jnp.array(list(map(int, args.architecture.split("-"))))
n_layers = len(layers)

weights = []
biases = []
for i in range(len(layers)-1):
    key, subkey1 = random.split(key, 2)
    weights.append(random.normal(subkey1, (layers[i+1], layers[i])) * jnp.sqrt(6/(layers[i] + layers[i+1])) * args.w_scale)
    biases.append(jnp.zeros(layers[i+1]))

opt_init, opt_update, get_params = optimizers.adam(args.l_rate)
opt_state = opt_init((weights, biases))

print(model_uuid)
print(args)
loss_hist = []
train_hist = []
test_hist = []
for e in range(args.epochs):
    acc_ta, acc_te, loss_r, s_te, s_ta = 0, 0, 0, 0, 0
    for x_train, y_train in train_dl:
        x_train = jnp.array(x_train.reshape((x_train.shape[0], x_train.shape[1], -1)))
        y_train = jnp.array(y_train)


        loss, opt_state, weights, biases = update_w(opt_state, get_params, opt_update, args.alpha, args.gamma, args.beta, args.thr, x_train, y_train, loss_fn, e)
        pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_train)
        
        acc_ta = (acc_ta * s_ta + float(acc_compute(pred, y_train))  * int(x_train.shape[0])) / (s_ta + int(x_train.shape[0]))
        loss_r = (loss_r * s_ta + loss * int(x_train.shape[0])) / (s_ta + int(x_train.shape[0]))
        s_ta += int(x_train.shape[0])
        print("{} {:.4f} {:.4f}".format(s_ta, loss_r, acc_ta))

    for x_test, y_test in test_dl:
        x_test = jnp.array(x_test.reshape((x_test.shape[0], x_test.shape[1], -1)))
        y_test = jnp.array(y_test)

        pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_test)
        
        acc_te = (acc_te * s_te + float(acc_compute(pred, y_test)) * int(x_test.shape[0])) / (s_te + int(x_test.shape[0]))
        s_te += int(x_test.shape[0])
        print("{} {:.4f}".format(s_te, acc_te))

    loss_hist.append(loss_r)
    train_hist.append(acc_ta)
    test_hist.append(acc_te)
    print("Epoch {} {:.4f} {:.4f} {:.4f}".format(e, loss_hist[-1], train_hist[-1], test_hist[-1]))

# Visualization
#pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_test)
#yy_plot(x_test, pred, model_uuid + '_yin_yang', str(loss_hist[-1]))
#pattern_plot(x_train[0,:,:], y_train[0,:,:], pred[0,:,:], model_uuid + "_pattern_visual", "")
curve_plot(loss_hist, train_hist, test_hist, model_uuid + "_curve", str(loss_hist[-1]))

# save model
jnp.savez("models/"+str(model_uuid)+".npz", weights = weights,  biases = biases, loss_hist = loss_hist, train_hist = train_hist, test_hist = test_hist,  args = args)
# add log entry
with open(args.log_file,'a') as f:
    f.write(str(loss_hist[-1]) + "," + str(train_hist[-1]) + "," + str(test_hist[-1]) + "," + model_uuid + "," + ",".join( [str(vars(args)[t]) for t in vars(args)]) + "," + str(datetime.datetime.now()) + "\n")

# # load model 
# model_uuid = 'abfcaa0f-1de3-492a-a08e-5fcce8da513c'
# npzfile = jnp.load("models/" + model_uuid + ".npz", allow_pickle=True)
# weights = npzfile['arr_0']
# biases = npzfile['arr_1']
# loss_hist = npzfile['arr_2']
# args = str(npzfile['arr_3'])[10:-1].split(',')
# dargs = {}
# for i in args:
#     var, val = i.split("=")
#     if "'" in val:
#         dargs[var.strip(" ")] = val.strip("'")
#     else:
#         dargs[var.strip(" ")] = float(val)
# args = argparse.Namespace(**dargs)

# pred = run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_train)
# loss_v = vr_loss(args.alpha_vr, pred, y_train)
# print(loss_v)
# curve_plot(loss_hist, model_uuid + "_curve", str(loss_v) + " " + str(args.alpha_vr))
# pattern_plot(x_train, y_train, pred, model_uuid + "_pattern_visual", str(loss_v) + " " + str(args.alpha_vr))