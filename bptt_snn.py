import argparse, time, pickle, uuid, os, datetime

from functools import partial
import jax.numpy as jnp
from jax.experimental import optimizers
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



parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--log-file", type=str, default="logs/test.csv", help='Log-file')
parser.add_argument("--seed", type=int, default=80085, help='Random seed')

parser.add_argument("--data-set", type=str, default="Yin_Yang", help='Data set to use')
parser.add_argument("--architecture", type=str, default="4-120-3", help='Architecture of the networks')

parser.add_argument("--l_rate", type=float, default=1e-3, help='Learning Rate')
parser.add_argument("--epochs", type=int, default=10, help='Epochs')

parser.add_argument("--w-scale", type=float, default=1., help='Weight Scaling')
parser.add_argument("--batch-size", type=float, default=128, help='Batch Size ')

parser.add_argument("--alpha", type=float, default=.6, help='Time constant for membrane potential')
parser.add_argument("--gamma", type=float, default=1.2, help='Reset Magnitude')
parser.add_argument("--thr", type=float, default=1., help='Membrane Threshold')

parser.add_argument("--alpha_vr", type=float, default=.85, help='Time constant for Van Rossum distance')

args = parser.parse_args()


key = random.PRNGKey(args.seed)
model_uuid = str(uuid.uuid4())

        

if args.data_set == 'Smile':
    if os.path.exists("data/smile_data_set/input_700_250_25.pkl") and os.path.exists("data/smile_data_set/smile95.pkl"):
        with open("data/smile_data_set/input_700_250_25.pkl", 'rb') as f:
            x_train = jnp.array(pickle.load(f)).transpose()
            x_train = x_train.reshape((1, x_train.shape[0], x_train.shape[1]))
        with open("data/smile_data_set/smile95.pkl", 'rb') as f:
            y_train = jnp.array(pickle.load(f))
            y_train = y_train.reshape((1, y_train.shape[0], y_train.shape[1]))
    else:
        raise Exception("Smile data set files do not exist please place them into data/smile_data_set")
    train_dl = [(x_train, y_train)]
    test_dl = [(x_train, y_train)]
    loss_fn = vr_loss
elif args.data_set == 'Yin_Yang':
    from data.yin_yang_data_set.dataset import YinYangDataset, to_spike_train
    from torch.utils.data import DataLoader

    dataset_train = YinYangDataset(size=5000, seed=42, transform=to_spike_train(100))
    dataset_validation = YinYangDataset(size=1000, seed=41, transform=to_spike_train(100))
    dataset_test = YinYangDataset(size=1000, seed=40, transform=to_spike_train(100))

    train_dl = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    loss_fn = nll_loss
elif args.data_set == 'NMNIST':
    # 2* 32* 32 = 2048 and 10 classes
    from torchneuromorphic.nmnist.nmnist_dataloaders import *
    import torchneuromorphic.transforms as transforms

    if os.path.exists('data/nmnist/n_mnist.hdf5'):
        pass
    elif os.path.exists('data/nmnist/'):
        out = create_events_hdf5('data/nmnist', 'data/nmnist/n_mnist.hdf5')
    else:
        raise Exception("NMNIST data set does not exist, download and place raw data into data/nmnist")
    
    train_dl, test_dl = create_dataloader(
            root='data/nmnist/n_mnist.hdf5',
            batch_size=args.batch_size,
            ds=1,
            num_workers=4)
    loss_fn = nll_loss
elif args.data_set == 'DVS_Gestures':
    from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
    import torchneuromorphic.transforms as transforms

    if os.path.exists('data/dvsgesture/dvs_gestures_build19.hdf5'):
        pass
    elif os.path.exists('data/dvsgesture/raw/'):
        out = create_events_hdf5('data/dvsgesture/raw/', 'data/dvsgesture/dvs_gestures_build19.hdf5')
    else:
        raise Exception("DVS Gestures data set does not exist, download and place raw data into data/dvsgesture/raw/")
    
    train_dl, test_dl = create_dataloader(
            root='data/dvsgesture/dvs_gestures_build19.hdf5',
            batch_size=args.batch_size,
            ds=4,
            num_workers=4)
    loss_fn = nll_loss
else:
    raise Exception("Unknown data set")



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
pred = v_run_snn(weights, biases, args.alpha, args.gamma, args.thr, x_train)
yy_plot(x_test, y_test, model_uuid + '_yin_yang', str(loss_hist[-1]))
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
