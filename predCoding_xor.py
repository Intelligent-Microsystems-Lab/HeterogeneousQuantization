from functools import partial
import jax.numpy as jnp
from jax import grad, jit, lax, vmap, value_and_grad, custom_vjp, random, device_put
import jax

from torchvision import datasets, transforms
import torch

@jit
def rmse(sout, pred):
    return jnp.sqrt(jnp.mean((sout - pred)**2))

#@jax.partial(jit, static_argnums=3)
def run_nn(weights, biases, sin, act_fn):
    x = sin
    for w,b in zip(weights, biases):
        x = jnp.dot(w, act_fn(x)) + b
    return x

v_run_nn = jit(vmap(run_nn, in_axes = (None, None, 0, None)), static_argnums=3)

#@jax.partial(jit, static_argnums=[3, 4, 5])
def infer_pc(x, weights, biases, act_fn, beta, it_max, var_layer):
    n_layers = len(weights) + 1 

    # infer function
    error = [[] for i in range(n_layers)]
    f_n = [[] for i in range(n_layers)]
    f_p = [[] for i in range(n_layers)]
    # calculate initial errors
    for i in range(1,n_layers):
        f_n[i-1] = act_fn(x[i-1])
        f_p[i-1] = vmap(grad(act_fn))(x[i-1])
        error[i] = (x[i] - jnp.dot(weights[i-1], f_n[i-1]) - biases[i-1])/var_layer[i]

    for i in range(it_max):
        # update variable nodes
        for l in range(1,n_layers-1):
            g = jnp.dot(weights[l].transpose(), error[l+1]) * f_p[l]
            x[l] = x[l] + beta * (-error[l] + g)
        # calculate errors
        for i in range(1,n_layers):
            f_n[i-1] = act_fn(x[i-1])
            f_p[i-1] = vmap(grad(act_fn))(x[i-1])
            error[i] = (x[i] - jnp.dot(weights[i-1], f_n[i-1]) - biases[i-1])/var_layer[i]
    return x, error

@jax.partial(jit, static_argnums=[4, 5, 6, 7])
def learn_pc(sin, sout, weights, biases, act_fn, l_rate, beta, it_max, var_layer):
    n_layers = len(weights)+1
    v_out = var_layer[-1]
    x = [[] for i in range(n_layers)]
    grad_w = [[] for i in weights]
    grad_b = [[] for i in biases]


    x[0] = sin
    # make predictions
    for i in range(1,n_layers):
        x[i] = jnp.dot(weights[i-1], act_fn(x[i-1])) + biases[i-1]

    # infer
    x[-1] = sout
    x, error = infer_pc(x, weights, biases, act_fn, beta, it_max, var_layer)

    # calculate gradients
    for i in range(n_layers-1):
        grad_b[i] = v_out * error[i+1]
        grad_w[i] = v_out * jnp.outer(error[i+1], act_fn(x[i]).transpose())
    return grad_w, grad_b

v_learn_pc = jit(vmap(learn_pc, in_axes = (0, 0, None, None, None, None, None, None, None)), static_argnums=[4, 5, 6, 7])

key = random.PRNGKey(80085)

# type relu
l_rate = .2
it_max = 100
epochs = 500
beta = .2 # euler integration constant
d_rate = .0 # weight decay

sin = jnp.array([[0.,0.,1.,1.], 
                 [0.,1.,0.,1.]]).transpose()
sout = jnp.array([[1.,0.,0.,1.]]).transpose()

act_fn = jax.numpy.tanh #jax.nn.relu

layers = jnp.array([2, 5, 1])
n_layers = len(layers)

var_layer = jnp.array([1]*(n_layers-1) + [10])

rmse_hist = []
for t in range(4):
    print("Trial {}".format(t))
    rmse_hist.append([])
    x_labels = []
    # init weights
    weights = []
    biases = []
    for i in range(len(layers)-1):
        key, subkey1 = random.split(key, 2)
        #weights.append(random.normal(subkey1, (layers[i+1], layers[i])) * jnp.sqrt(6/layers[i]))
        weights.append(random.uniform(subkey1, (layers[i+1], layers[i]), minval=-1.0, maxval=1.0) * jnp.sqrt(6/(layers[i] + layers[i+1])))
        biases.append(jnp.zeros(layers[i+1]))

    # test
    weights = [jnp.array([[1,1], [1,1], [1,1], [1,1], [1,1]]), jnp.array([[1,1,1,1,1]])]

    for e in range(epochs):
        if e%50 == 0:
            pred_new = v_run_nn(weights, biases, sin, act_fn)
            print("{:4d} {:.4f}".format(e, rmse(sout, pred_new)))
            rmse_hist[t].append(rmse(sout, pred_new))
            x_labels.append(e)

        for x, y in zip(sin, sout):
            grad_w, grad_b = learn_pc(x, y, weights, biases, act_fn, l_rate, beta, it_max, var_layer)
            weights = [w + l_rate * gw for w, gw in zip(weights, grad_w)]
            biases = [b + l_rate * gb for b, gb in zip(biases, grad_b)]


    pred_new = v_run_nn(weights, biases, sin, act_fn)
    rmse_hist[t].append(rmse(sout, pred_new))
    x_labels.append(e)
    print("{:4d} {:.4f}".format(e, rmse(sout, pred_new)))



import matplotlib.pyplot as plt

for i, err in enumerate(rmse_hist):
    plt.plot(x_labels, err, label = "Run {}".format(i+1))

plt.title("Predictive Coding - XOR")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()

plt.tight_layout()
plt.savefig('figures/xor_rmse.png')
plt.close()


