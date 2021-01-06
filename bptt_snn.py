import argparse, time, pickle, uuid, os, datetime

from functools import partial
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax

import matplotlib.pyplot as plt

from datasets import dl_create
from visualization import pattern_plot, yy_plot, curve_plot
from snn_util import v_run_snn, update_w

# module load python cuda/10.2
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--log-file", type=str, default="logs/test.csv", help='Log-file')
parser.add_argument("--seed", type=int, default=80085, help='Random seed')

parser.add_argument("--data-set", type=str, default="Yin_Yang", help='Data set to use')
parser.add_argument("--architecture", type=str, default="4-120-3", help='Architecture of the networks')

parser.add_argument("--l_rate", type=float, default=1e-3, help='Learning Rate')
parser.add_argument("--epochs", type=int, default=10000, help='Epochs')

parser.add_argument("--w-scale", type=float, default=2., help='Weight Scaling')
parser.add_argument("--batch-size", type=float, default=128, help='Batch Size ')

parser.add_argument("--alpha", type=float, default=.95, help='Time constant for membrane potential')
parser.add_argument("--gamma", type=float, default=1.2, help='Reset Magnitude')
parser.add_argument("--thr", type=float, default=1., help='Membrane Threshold')

parser.add_argument("--alpha_vr", type=float, default=.85, help='Time constant for Van Rossum distance')

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

        loss, opt_state, weights, biases = update_w(opt_state, get_params, opt_update, args.alpha, args.gamma, args.alpha_vr, args.thr, x_train, y_train, loss_fn, e)
        pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_train)
        
        acc_ta = (acc_ta * s_ta + float(acc_compute(pred, y_train))  * int(x_train.shape[0])) / (s_ta + int(x_train.shape[0]))
        loss_r = (loss_r * s_ta + loss * int(x_train.shape[0])) / (s_ta + int(x_train.shape[0]))
        s_ta += int(x_train.shape[0])
        #print("{} {:.4f} {:.4f}".format(s_ta, loss_r, acc_ta))

    for x_test, y_test in test_dl:
        x_test = jnp.array(x_test.reshape((x_test.shape[0], x_test.shape[1], -1)))
        y_test = jnp.array(y_test)

        pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_test)
        
        acc_te = (acc_te * s_te + float(acc_compute(pred, y_test)) * int(x_test.shape[0])) / (s_te + int(x_test.shape[0]))
        s_te += int(x_test.shape[0])
        #print("{} {:.4f}".format(s_te, acc_te))


    loss_hist.append(loss_r)
    train_hist.append(acc_ta)
    test_hist.append(acc_te)
    print("Epoch {} {:.4f} {:.4f} {:.4f}".format(e, loss_hist[-1], train_hist[-1], test_hist[-1]))

# Visualization
pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_test)
yy_plot(x_test, pred, model_uuid + '_yin_yang', str(loss_hist[-1]))
# pattern_plot(x_train[0,:,:], y_train[0,:,:], pred[0,:,:], model_uuid + "_pattern_visual", "")
curve_plot(loss_hist, train_hist, test_hist, model_uuid + "_curve", str(loss_hist[-1]) + " " + str(args.alpha_vr))

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
