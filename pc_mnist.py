from functools import partial
import jax.numpy as jnp
from jax import (
    grad,
    jit,
    lax,
    vmap,
    value_and_grad,
    custom_vjp,
    random,
    device_put,
)
import jax

from torchvision import datasets, transforms
import torch

# module load python cuda/10.2
# export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"


@jit
def rmse(sout, pred):
    return jnp.sqrt(jnp.mean((sout - pred) ** 2))


@jax.partial(jit, static_argnums=3)
def run_nn(weights, biases, sin, act_fn):
    x = sin
    for w, b in zip(weights, biases):
        x = jnp.dot(w, act_fn(x)) + b
    return x


v_run_nn = jit(vmap(run_nn, in_axes=(None, None, 0, None)), static_argnums=3)


@jax.partial(jit, static_argnums=[3, 4, 5])
def infer_pc(x, weights, biases, act_fn, beta, it_max, var_layer):
    n_layers = len(weights) + 1

    # infer function
    error = [[] for i in range(n_layers)]
    f_n = [[] for i in range(n_layers)]
    f_p = [[] for i in range(n_layers)]
    # calculate initial errors
    for i in range(1, n_layers):
        f_n[i - 1] = act_fn(x[i - 1])
        f_p[i - 1] = vmap(grad(act_fn))(x[i - 1])
        error[i] = (
            x[i] - jnp.dot(weights[i - 1], f_n[i - 1]) - biases[i - 1]
        ) / var_layer[i]

    for i in range(it_max):
        # update variable nodes
        for l in range(1, n_layers - 1):
            g = jnp.dot(weights[l].transpose(), error[l + 1]) * f_p[l]
            x[l] = x[l] + beta * (-error[l] + g)
        # calculate errors
        for i in range(1, n_layers):
            f_n[i - 1] = act_fn(x[i - 1])
            f_p[i - 1] = vmap(grad(act_fn))(x[i - 1])
            error[i] = (
                x[i] - jnp.dot(weights[i - 1], f_n[i - 1]) - biases[i - 1]
            ) / var_layer[i]
    return x, error


@jax.partial(jit, static_argnums=[4, 5, 6, 7])
def learn_pc(
    sin, sout, weights, biases, act_fn, l_rate, beta, it_max, var_layer
):
    n_layers = len(weights) + 1
    v_out = var_layer[-1]
    x = [[] for i in range(n_layers)]
    grad_w = [[] for i in weights]
    grad_b = [[] for i in biases]

    x[0] = sin
    # make predictions
    for i in range(1, n_layers):
        x[i] = jnp.dot(weights[i - 1], act_fn(x[i - 1])) + biases[i - 1]

    # infer
    x[-1] = sout
    x, error = infer_pc(x, weights, biases, act_fn, beta, it_max, var_layer)

    # calculate gradients
    for i in range(n_layers - 1):
        grad_b[i] = v_out * error[i + 1]
        grad_w[i] = v_out * jnp.outer(error[i + 1], act_fn(x[i]).transpose())
    return grad_w, grad_b


v_learn_pc = jit(
    vmap(learn_pc, in_axes=(0, 0, None, None, None, None, None, None, None)),
    static_argnums=[4, 5, 6, 7],
)

key = random.PRNGKey(80085)
batch_size = 512

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
dataset1 = datasets.MNIST(
    "data", train=True, download=True, transform=transform
)
dataset2 = datasets.MNIST("data", train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)


l_rate = 0.0001
it_max = 100
epochs = 50
beta = 0.02  # euler integration constant

act_fn = jax.numpy.tanh  # jax.nn.relu

layers = jnp.array([784, 500, 100, 10])
n_layers = len(layers)

var_layer = jnp.array([1] * (n_layers))


weights = []
biases = []
for i in range(len(layers) - 1):
    key, subkey1 = random.split(key, 2)
    weights.append(
        random.normal(subkey1, (layers[i + 1], layers[i]))
        * jnp.sqrt(6 / (layers[i] + layers[i + 1]))
    )
    # weights.append(random.uniform(subkey1, (layers[i+1], layers[i]), minval=-1.0, maxval=1.0) * jnp.sqrt(6/(layers[i] + layers[i+1])))
    biases.append(jnp.zeros(layers[i + 1]))

train_acc = []
test_acc = []
for e in range(epochs):
    run_acc = jnp.array([])
    for data_x, data_l in train_loader:
        data_x = jnp.array(data_x.flatten(1))
        data_y = jnp.array(torch.nn.functional.one_hot(data_l))

        grad_w, grad_b = v_learn_pc(
            data_x,
            data_y,
            weights,
            biases,
            act_fn,
            l_rate,
            beta,
            it_max,
            var_layer,
        )
        weights = [w + l_rate * gw.sum(0) for w, gw in zip(weights, grad_w)]
        biases = [b + l_rate * gb.sum(0) for b, gb in zip(biases, grad_b)]

        pred_new = v_run_nn(weights, biases, data_x, act_fn)
        run_acc = jnp.concatenate(
            (run_acc, pred_new.argmax(1) == jnp.array(data_l))
        )
    train_acc.append(run_acc.mean())

    run_acc = jnp.array([])
    for data_x, data_l in test_loader:
        data_x = jnp.array(data_x.flatten(1))
        data_y = jnp.array(torch.nn.functional.one_hot(data_l))

        pred_new = v_run_nn(weights, biases, data_x, act_fn)
        run_acc = jnp.concatenate(
            (run_acc, pred_new.argmax(1) == jnp.array(data_l))
        )
    test_acc.append(run_acc.mean())

    print("{:4d} {:.4f} {:.4f}".format(e, train_acc[-1], test_acc[-1]))


import matplotlib.pyplot as plt

plt.plot(train_acc, label="Training")
plt.plot(test_acc, label="Testing")

plt.title("Predictive Coding - MNIST")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("figures/pc_mnist.png")
plt.close()
