# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer

import pickle
import time
import datetime
import sys

from absl import app
from absl import flags
from absl import logging
import functools

import jax
import jax.numpy as jnp


from flax.metrics import tensorboard
import matplotlib.pyplot as plt

sys.path.append("../..")
from model import init_state, nn_model, init_params  # noqa: E402

# from jax.config import config
# config.update('jax_disable_jit', True)

# parameters
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/smile_bptt-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)
INPUT_FILE = flags.DEFINE_string(
    "input_file", "../../datasets/smile/input_700_250_25.pkl", ""
)
TARGET_FILE = flags.DEFINE_string(
    "target_file", "../../datasets/smile/smile95.pkl", ""
)
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.5, "")
INIT_SCALE_S = flags.DEFINE_float("init_scale_s", 0.1, "")
TRAINING_STEPS = flags.DEFINE_integer("training_steps", 10_000, "")
HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 64, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 10, "")
EVALUATION_NOISE = flags.DEFINE_float("evaluation_noise", 0.05, "")
SEQ_LEN = flags.DEFINE_integer("seq_len", 50, "")
SEED = flags.DEFINE_integer("seed", 42, "")

DTYPE = jnp.float32


def mse_loss(logits, labels):
  # simple MSE loss
  loss = jnp.mean((logits[:, 0, :] - labels[:, 0, :]) ** 2)

  return loss


@jax.jit
def train_step(params, batch):
  out_dim = batch["target_seq"].shape[1]

  init_s = init_state(out_dim, 1, HIDDEN_SIZE.value, DTYPE)

  def loss_fn(params):
    nn_model_fn = functools.partial(nn_model, params)
    final_carry, output_seq = jax.lax.scan(
        nn_model_fn, init=init_s, xs=batch["input_seq"]
    )
    loss = mse_loss(output_seq, batch["target_seq"])
    return loss, output_seq

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss_val, logits), grad = grad_fn(params)

  # simple SGD step
  params = jax.tree_multimap(
      lambda x, y: x - LEARNING_RATE.value * y, params, grad
  )

  return params, loss_val, logits


@jax.jit
def eval_model(params, batch, rng):
  out_dim = batch["target_seq"].shape[1]

  init_s = init_state(out_dim, 1, HIDDEN_SIZE.value, DTYPE)

  k1, k2, k3 = jax.random.split(rng, 3)
  noise_tree = {
      "cf": {
          "h1": jax.random.normal(k1, params["cf"]["h1"].shape)
          * jnp.max(jnp.abs(params["cf"]["h1"]))
          * EVALUATION_NOISE.value,
          "w1": jax.random.normal(k2, params["cf"]["w1"].shape)
          * jnp.max(jnp.abs(params["cf"]["w1"]))
          * EVALUATION_NOISE.value,
      },
      "of": {
          "wo": jax.random.normal(k3, params["of"]["wo"].shape)
          * jnp.max(jnp.abs(params["of"]["wo"]))
          * EVALUATION_NOISE.value,
      },
  }
  params_noise = jax.tree_multimap(lambda x, y: x + y, params, noise_tree)

  nn_model_fn = functools.partial(nn_model, params_noise)
  final_carry, output_seq = jax.lax.scan(
      nn_model_fn, init=init_s, xs=batch["input_seq"]
  )
  loss = mse_loss(output_seq, batch["target_seq"])

  return loss


def get_data():
  """Load smile dataset into memory."""
  with open(INPUT_FILE.value, "rb") as f:
    x_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

  with open(TARGET_FILE.value, "rb") as f:
    y_train = jnp.expand_dims(jnp.array(pickle.load(f)).transpose(), 0)

  x_train = jax.image.resize(
      x_train,
      (x_train.shape[0], SEQ_LEN.value, x_train.shape[2]),
      method="linear",
  )
  y_train = jax.image.resize(
      y_train,
      (y_train.shape[0], SEQ_LEN.value, SEQ_LEN.value),
      method="linear",
  )

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

  # initialize parameters
  rng, p_rng = jax.random.split(rng, 2)
  params = init_params(
      p_rng, inp_dim, out_dim, INIT_SCALE_S.value, HIDDEN_SIZE.value
  )

  # Training loop.
  logging.info("Files in: " + WORK_DIR.value)
  logging.info(jax.devices())

  t_loop_start = time.time()
  for step in range(TRAINING_STEPS.value):
    rng, p_rng = jax.random.split(rng, 2)
    params, loss_val, logits = train_step(params, data)

    summary_writer.scalar("train_loss", loss_val, (step + 1))
    summary_writer.scalar(
        "step_time", (time.time() - t_loop_start), (step + 1)
    )
    summary_writer.scalar(
        "eval_loss", eval_model(params, data, p_rng), (step + 1)
    )
    t_loop_start = time.time()

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
