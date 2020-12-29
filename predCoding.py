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

#@jax.partial(jit, static_argnums=[4, 5, 6, 7])
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

var_layer = jnp.array([1]*(n_layers-1) + [1])

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
    #weights = [jnp.array([[1,1], [1,1], [1,1], [1,1], [1,1]]), jnp.array([[1,1,1,1,1]])]

    for e in range(epochs):
        if e%50 == 0:
            pred_new = v_run_nn(weights, biases, sin, act_fn)
            print("{:4d} {:.4f}".format(e, rmse(sout, pred_new)))
            rmse_hist[t].append(rmse(sout, pred_new))
            x_labels.append(e)

        for x, y in zip(sin, sout):
            grad_w, grad_b = v_learn_pc(x, y, weights, biases, act_fn, l_rate, beta, it_max, var_layer)
            weights = [w + l_rate * gw for w, gw in zip(weights, grad_w)]
            biases = [b + l_rate * gb for b, gb in zip(biases, grad_b)]


    pred_new = v_run_nn(weights, biases, sin, act_fn)
    rmse_hist[t].append(rmse(sout, pred_new))
    x_labels.append(e)
    print("{:4d} {:.4f}".format(e, rmse(sout, pred_new)))



import matplotlib.pyplot as plt

for i, err in enumerate(rmse_hist):
    plt.plot(x_labels, err, label = "Run {}".format(i+1))

plt.title("Predictive Coding")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.legend()

plt.tight_layout()
plt.savefig('figures/xor_rmse.png')
plt.close()






batch_size = 256

transform=transforms.Compose([
    transforms.ToTensor(),
    ])
dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size = batch_size)


# simple setup 
data_x, data_y = next(iter(train_loader))

# .unsqueeze(dim=0)
data_x, data_y = jnp.array(data_x[0].flatten()), jnp.array(torch.nn.functional.one_hot(data_y[0], num_classes=10))

# # pytorch code
# def forward(self,x):
#     self.inp = x.clone()
#     self.activations = torch.matmul(self.inp, self.weights)
#     return self.f(self.activations)

# def backward(self,e):
#     self.fn_deriv = self.df(self.activations)
#     out = torch.matmul(e * self.fn_deriv, self.weights.T)
#     return torch.clamp(out,-50,50)

# def update_weights(self,e,update_weights=False,sign_reverse=False):
#     self.fn_deriv = self.df(self.activations)
#     dw = torch.matmul(self.inp.T, e * self.fn_deriv)
#     if update_weights:
#       if sign_reverse==True:
#         self.weights -= self.learning_rate * torch.clamp(dw*2,-50,50)
#       else:
#         self.weights += self.learning_rate * torch.clamp(dw*2,-50,50)
#     return dw


layers = [784, 500, 300, 100, 10] #[10, 100, 300, 500, 784] # "reverse order"
T = 100
gamma = .001

act_fn = jax.nn.relu
def grad_fn(x):
    return (x >= 0).astype(jnp.float32)

# init weights
weights = []
for i in range(len(layers)-1):
    key, subkey1 = random.split(key, 2)
    weights.append(random.normal(subkey1, (layers[i], layers[i+1])) * jnp.sqrt(6/layers[i]))

mu = [[] for i in range(len(layers))]
out = [[] for i in range(len(layers))]
pred = [[] for i in range(len(layers))]
pred_error = [[] for i in range(len(layers))]

mu[0] = data_x
out[0] = data_x

for l, w in enumerate(weights):
    mu[l+1] = act_fn(jnp.dot(mu[l], weights[l]))
    out[l+1] = act_fn(jnp.dot(mu[l], weights[l]))
mu[-1] = data_y

pred_error[-1] = mu[-1] - out[-1] 
pred[-1] = pred_error[-1]

for t in range(T):
    for l in reversed(range(len(layers)-1)):
        if l != 0:
            pred_error[l] = mu[l] - out[l]
            pred[l] = jnp.dot(pred_error[l+1] * grad_fn(out[l+1]) , weights[l].transpose())
            dx_l = pred_error[l] - pred[l]
            mu[l] -= gamma * dx_l



def run_pred_coding(weights, data_x, data_y, gamma, pred_mode, layers, T, act_fn, grad_fn, epsilon):
    x_t = []
    e_t = []

    # zero init x_t
    for i in layers:
        x_t.append(jnp.zeros((i)))
        e_t.append(jnp.zeros((i)))

    # set input and output
    if pred_mode:
        x_t[0] = jnp.zeros_like(data_y)
    else:
        x_t[0] = data_y
    x_t[-1] = data_x

    delta_w = []
    chg_hist = []
    x_hist = []
    for t in range(T):
        chg_norm = 0
        x_norm = 0
        for l in reversed(range(len(layers)-1)): # can be parallel
            mu_t = jnp.dot(act_fn(x_t[l+1]), weights[l].transpose())
            e_t[l]  = x_t[l] - mu_t 
            if l == 0 and pred_mode:
                delta_x = gamma * (-1. * e_t[l])
            elif l == 0 and not pred_mode:
                delta_x = jnp.zeros_like(data_y)
            else:
                delta_x = gamma * (-1. * e_t[l] + grad_fn(x_t[l]) * jnp.dot(weights[l-1].transpose(), e_t[l-1]) ) 
            x_t[l] = x_t[l] + delta_x
            chg_norm += jnp.sum(jnp.abs(delta_x))
            x_norm += jnp.sum(jnp.abs(x_t[l]))
            if t == l and not pred_mode:
                delta_w.append(jnp.outer(e_t[l], act_fn(x_t[l+1])))
        chg_hist.append(chg_norm)
        x_hist.append(x_norm)
    return x_t[0], delta_w, chg_hist, x_hist

pred_coding_v = vmap(run_pred_coding, in_axes=(None, 0, 0, None, None, None, None, None, None, None))

def acc_comp(x_out, data_l):
    return jnp.mean((jnp.argmax(x_out, 1) == jnp.array(data_l)))


key = random.PRNGKey(80085)
batch_size = 256

transform=transforms.Compose([
    transforms.ToTensor(),
    ])
dataset1 = datasets.MNIST('data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size = batch_size)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size = batch_size)


layers = [10, 100, 300, 500, 784] # "reverse order"
T = 10000
epochs = 2
gamma = .03
alpha = .0001
pred_mode = True
epsilon = 10

act_fn = jax.nn.relu
def grad_fn(x):
    return (x >= 0).astype(jnp.float32)

# init weights
weights = []
for i in range(len(layers)-1):
    key, subkey1 = random.split(key, 2)
    weights.append(random.normal(subkey1, (layers[i], layers[i+1])))

for e in range(epochs):
    for data_x, data_l in train_loader:
        data_x = jnp.array(data_x.flatten(1))
        data_y = jnp.array(torch.nn.functional.one_hot(data_l))
        #print(jnp.sum(jnp.array([np.sum(cur**2)  for cur in weights  ])))
        x_out, dw, chg_hist, x_hist = pred_coding_v(weights, data_x, data_y, gamma, False, layers, 10000, act_fn, grad_fn, epsilon)
        weights = [w + alpha * d_w.sum(0)/d_w.shape[0] for w, d_w in zip(weights, dw)]
        x_out, _, chg_hist, x_hist = pred_coding_v(weights, data_x, data_y, gamma, True, layers, 10000, act_fn, grad_fn, epsilon)
        
        print("{:.4f} {:.4f} {:.4f}".format(acc_comp(x_out, data_l), x_out.max(), x_out.min()))
        break


import matplotlib.pyplot as plt


# for i in range(10):
#     x_p = []
#     for t in range(T):
#         x_p.append(chg_hist[t][i])
#     plt.plot(x_p)

# plt.tight_layout()
# plt.savefig('figures/mult_curves.png')
# plt.close()


for i in range(10):
    x_p = []
    for t in range(T):
        x_p.append(x_hist[t][i])
    plt.plot(x_p)

plt.tight_layout()
plt.savefig('figures/mult_x_curves.png')
plt.close()


# for i in range(1):
#     x_p = []
#     for t in range(T):
#         x_p.append(chg_hist[t][i])
#     plt.plot(x_p)

# plt.tight_layout()
# plt.savefig('figures/one_curves.png')
# plt.close()
