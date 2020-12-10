import argparse, time, pickle
from functools import partial

import jax.numpy as jnp
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
from jax.experimental import optimizers, stax

import matplotlib.pyplot as plt


# module load python cuda/10.2
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--input", type=str, default="./data/input_700_250_25.pkl", help='Input pickle')
parser.add_argument("--target", type=str, default="./data/smile95.pkl", help='Target pattern pickle')

parser.add_argument("--time_step", type=float, default=1e-3, help='Simulation time step size')
parser.add_argument("--nb_steps", type=float, default=250, help='Simulation steps')
parser.add_argument("--nb_epochs", type=float, default=10000, help='Simulation steps')

parser.add_argument("--tau_mem", type=float, default=10e-3, help='Time constant for membrane potential')
parser.add_argument("--tau_syn", type=float, default=5e-3, help='Time constant for synapse')
parser.add_argument("--tau_vr", type=float, default=5e-3, help='Time constant for Van Rossum distance')
parser.add_argument("--alpha", type=float, default=.75, help='Time constant for synapse')
parser.add_argument("--beta", type=float, default=.875, help='Time constant for Van Rossum distance')

args = parser.parse_args()


def update_w_gc(i, opt_state, opt_update, get_params, params_fixed, x_bxt, f_bxt):
    params = get_params(opt_state)  
    grads = grad(vr_loss, argnums=1)(params_fixed, params, x_bxt, f_bxt)
    return opt_update(i, grads, opt_state)

update_w_gc_jit = jit(update_w_gc, static_argnums=(2,3)) #update_w_gc # 
loss_jit = jit(vr_loss) #vr_loss #


with open("data/input_700_250_25.pkl", 'rb') as f:
    x_train = jnp.array(pickle.load(f)).transpose()
with open("data/smile95.pkl", 'rb') as f:
    y_train = jnp.array(pickle.load(f))

key = random.PRNGKey(80085)

t_steps     = 250
in_size     = 700
hidden_size = 400
out_size    = 250

params_fixed = {'alpha_vr' : jnp.array([ .75]),
                'alpha'    : jnp.array([ .75]),
                'beta'     : jnp.array([ .875]),
                'thr'      : jnp.array([ .9]),
                'gamma'    : jnp.array([0.]),
                'delta'    : jnp.array([1.])}

key, subkey1, subkey2 = random.split(key, 3)
params = [{'w' : random.normal(subkey1, (hidden_size, in_size))}, 
          {'w' : random.normal(subkey2, (out_size, hidden_size))}]

step_size = .1            # initial learning rate
decay_factor = .5        # decay the learning rate this much
adam_b1 = 0.9             # Adam parameters
adam_b2 = 0.999
adam_eps = 1e-1

epochs = 10000

decay_fun = optimizers.exponential_decay(step_size, decay_steps=2000, decay_rate=decay_factor)
opt_init, opt_update, get_params = optimizers.adam(decay_fun, adam_b1, adam_b2, adam_eps)
opt_state = opt_init(params)


# Run the optimization loop, first jit'd call will take a minute.
start_time = time.time()
all_train_losses = []
for batch in range(epochs):
    opt_state = update_w_gc_jit(batch, opt_state, opt_update, get_params, params_fixed, x_train, y_train)

    if batch % 100 == 0:
        params = get_params(opt_state)
        all_train_losses.append(loss_jit(params_fixed, params, x_train, y_train))
        train_loss = all_train_losses[-1]
        batch_time = time.time() - start_time
        step_size = decay_fun(batch)
        s = "Batch {} in {:0.2f} sec, step size: {:0.5f}, training loss {:0.4f}"
        print(s.format(batch, batch_time, step_size, train_loss))
        start_time = time.time()

params = get_params(opt_state)
state0 = state_init(params)
pred = snn_net_run(params_fixed, params, x_train, state0)
plot_helper(x_train.transpose(), pred, y_train)








