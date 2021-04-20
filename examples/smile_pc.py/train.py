# copied from
# https://github.com/deepmind/dm-haiku/blob/master/examples/rnn/train.py
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Character-level language modelling with a recurrent network in JAX."""

from typing import Any, NamedTuple
import pickle
import time

from absl import app
from absl import flags
from absl import logging
import functools

import jax
import jax.numpy as jnp
import optax
import sys

import tensorflow as tf
import tensorflow_datasets as tfds
from flax.metrics import tensorboard

import matplotlib.pyplot as plt

import uuid

sys.path.append("../..")
from model import *

# from jax.config import config
# config.update('jax_disable_jit', True)

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir", "../../../training_dir/params_pc_smile/", ""
)
INPUT_FILE = flags.DEFINE_string(
    "input_file", "../../datasets/smile/input_700_250_25.pkl", ""
)
TARGET_FILE = flags.DEFINE_string(
    "target_file", "../../datasets/smile/smile95.pkl", ""
)
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.1, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 1000, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 1, "")
SEED = flags.DEFINE_integer("seed", 42, "")


def mse_loss(logits, labels):
    # simple MSE loss
    loss = jnp.mean((logits[:, 0, :] - labels[:, 0, :]) ** 2)

    return loss


def forward_sweep(input_seq, params):

    out_dim = input_seq.shape[1]
    init_s = init_state(out_dim, 1)

    out_pred = []
    h_pred = [init_s]

    p_core_fn = functools.partial(core_fn, params["cf"])
    p_out_fn = functools.partial(output_fn, params["of"])

    _, h_pred = jax.lax.scan(
        p_core_fn,
        init=init_s,
        xs=input_seq,
    )

    _, out_pred = jax.lax.scan(
        p_out_fn,
        init=jnp.zeros((input_seq.shape[1], params["of"]["bo"].shape[0])),
        xs=h_pred,
    )

    return out_pred, h_pred


n_inference_steps = 100


def relu_deriv(x):
    xrel = jax.nn.relu(x)
    return jax.lax.select(xrel > 0, jnp.ones_like(x), jnp.zeros_like(x))


def infer(params, input_seq, target_seq, y_pred, h_pred):
    out_dim = input_seq.shape[1]
    init_s = init_state(out_dim, 1)

    e_ys = [[] for i in range(len(target_seq))]  # ouptut prediction errors
    e_hs = [
        [] for i in range(len(input_seq))
    ]  # hidden state prediction errors

    hs = [init_s] + [x for x in h_pred]

    for i, (inp, targ) in reversed(
        list(enumerate(zip(input_seq, target_seq)))
    ):
        for n in range(n_inference_steps):
            e_ys[i] = targ - y_pred[i]
            e_hs[i] = hs[i + 1] - h_pred[i + 1]
            hdelta = e_hs[i] - jnp.dot(
                e_ys[i]
                * relu_deriv(jnp.dot(h_pred[i + 1], params["of"]["wo"])),
                params["of"]["wo"].transpose(),
            )
            if i < len(target_seq) - 1:
                fn_deriv = relu_deriv(
                    jnp.dot(h_pred[i + 1], params["cf"]["h1"])
                    + jnp.dot(input_seq[i + 1], params["cf"]["w1"])
                )
                hdelta -= jnp.dot((e_hs[i + 1] * fn_deriv), params["cf"]["h1"])
            hs[i + 1] -= 0.1 * hdelta

    return e_ys, e_hs


def compute_grads(params, input_seq, e_ys, e_hs, h_pred):
    dWy = jnp.zeros_like(params["of"]["wo"])
    dWx = jnp.zeros_like(params["cf"]["w1"])
    dWh = jnp.zeros_like(params["cf"]["h1"])
    for i in reversed(list(range(len(input_seq)))):
        fn_deriv = relu_deriv(
            jnp.dot(input_seq[i], params["cf"]["w1"])
            + jnp.dot(h_pred[i], params["cf"]["h1"])
        )
        dWy += jnp.dot(
            h_pred[i + 1].transpose(),
            (e_ys[i] * relu_deriv(jnp.dot(h_pred[i + 1], params["of"]["wo"]))),
        )
        dWx += jnp.dot(input_seq[i].transpose(), (e_hs[i] * fn_deriv))
        dWh += jnp.dot(h_pred[i].transpose(), (e_hs[i] * fn_deriv))

    return {
        "cf": {
            "w1": jnp.clip(dWx, -50, 50),
            "h1": jnp.clip(dWh, -50, 50),
            "b1": jnp.zeros_like(params["cf"]["b1"]),
        },
        "of": {
            "wo": jnp.clip(dWy, -50, 50),
            "bo": jnp.zeros_like(params["of"]["bo"]),
        },
    }


@jax.jit
def train_step(params, batch):
    import pdb

    pdb.set_trace()

    seq_len = batch["input_seq"].shape[0]

    out_pred, h_pred = forward_sweep(batch["input_seq"], params)
    e_ys, e_hs = infer(
        params, batch["input_seq"], batch["target_seq"], out_pred, h_pred
    )
    grad = compute_grads(params, batch["input_seq"], e_ys, e_hs, h_pred)

    # simple SGD step
    params = jax.tree_multimap(
        lambda x, y: x - LEARNING_RATE.value * y / seq_len, params, grad
    )

    return params, jnp.sum(jnp.abs(jnp.stack(e_ys))), out_pred


def get_data():
    """Load smile dataset into memory."""
    with open(INPUT_FILE.value, "rb") as f:
        x_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

    with open(TARGET_FILE.value, "rb") as f:
        y_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

    x_train = jnp.moveaxis(x_train, (0, 1, 2), (1, 0, 2))
    y_train = jnp.moveaxis(y_train, (0, 1, 2), (1, 0, 2))

    return {"input_seq": x_train, "target_seq": y_train}


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # get data set
    rng = jax.random.PRNGKey(SEED.value)
    data = get_data()

    inp_dim = data["input_seq"].shape[2]
    out_dim = data["target_seq"].shape[2]
    t_dim = data["input_seq"].shape[1]

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(p_rng, inp_dim, out_dim, INIT_SCALE_S.value)

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())

    params_hist = {0: params}
    t_loop_start = time.time()
    for step in range(TRAINING_STEPS.value):
        params, loss_val, logits = train_step(params, data)
        summary_writer.scalar("train_loss", loss_val, (step + 1))
        summary_writer.scalar(
            "step_time", (time.time() - t_loop_start), (step + 1)
        )
        t_loop_start = time.time()
        params_hist[step + 1] = params

        with open(WORK_DIR.value + "/params_hist.pickle", "wb") as handle:
            pickle.dump(params_hist, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Periodically report loss and show an example
        if (step + 1) % EVALUATION_INTERVAL.value == 0:
            logging.info("step: %d, loss: %.4f", step + 1, loss_val)

            # save picture
            plt.clf()
            fig, axes = plt.subplots(nrows=1, ncols=3)
            axes[0].matshow(data["target_seq"][:, 0, :].transpose())
            axes[1].matshow(logits[:, 0, :].transpose())
            axes[2].matshow(
                data["target_seq"][:, 0, :].transpose()
                - logits[:, 0, :].transpose()
            )
            plt.tight_layout()
            plt.savefig(WORK_DIR.value + "/smile.png")
            plt.close()


if __name__ == "__main__":
    app.run(main)
