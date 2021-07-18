# IMSL Lab - University of Notre Dame
# Author: Clemens JS Schaefer
# Dataset from https://arxiv.org/abs/1804.03209

from absl import app
from absl import flags
from absl import logging
from itertools import islice
import functools
import time
import datetime

import jax
import jax.numpy as jnp

import sys

from flax.metrics import tensorboard
from flax.training import common_utils
from flax import optim
from flax.training.lr_schedule import create_cosine_learning_rate_schedule

from kws_streaming.layers import speech_features
from kws_streaming.models import model_flags
from kws_streaming.data import input_data
from kws_streaming.layers import modes
from kws_streaming.train import base_parser
import tensorflow.compat.v1 as tf

from jax.config import config

config.update("jax_debug_nans", True)
# config.update("jax_disable_jit", True)

sys.path.append("../..")
from model import init_state, init_params, nn_model  # noqa: E402
from predictive_coding import forward_sweep, infer, compute_grads  # noqa: E402

# parameters
SEED = flags.DEFINE_integer("seed", 42, "")
WORK_DIR = flags.DEFINE_string(
    "work_dir",
    "../../../training_dir/kws_bptt-{date:%Y-%m-%d_%H-%M-%S}/".format(
        date=datetime.datetime.now()
    ),
    "",
)

TRAINING_STEPS = flags.DEFINE_integer("training_epochs", 23438, "")
WARMUP_EPOCHS = flags.DEFINE_integer("warmup_epochs", 10, "")
EVALUATION_INTERVAL = flags.DEFINE_integer("evaluation_interval", 1, "")  # 72

HIDDEN_SIZE = flags.DEFINE_integer("hidden_size", 1024, "")

BATCH_SIZE = flags.DEFINE_integer("batch_size", 1024, "")
LEARNING_RATE = flags.DEFINE_float("learning_rate", 0.001, "")
MOMENTUM = flags.DEFINE_float("momentum", 0.9, "")
LABEL_SMOOTHING = flags.DEFINE_float("label_smoothing", 0.1, "")
INFERENCE_STEPS = flags.DEFINE_integer("inference_steps", 50, "")
INFERENCE_LR = flags.DEFINE_float("inference_lr", 0.1, "")
# UPDATE_FREQ = flags.DEFINE_integer("update_freq", 100, "")
# GRAD_ACCUMULATE = flags.DEFINE_bool("grad_accumulate", True, "")
# GRAD_CLIP = flags.DEFINE_float("grad_clip", 100.0, "")

parser = base_parser.base_parser()
flags_input, unparsed = parser.parse_known_args()
flags_input = model_flags.update_flags(flags_input)


flags_input.mel_upper_edge_hertz = 7600
flags_input.window_size_ms = 30
flags_input.window_stride_ms = 10
flags_input.mel_num_bins = 80
flags_input.dct_num_features = 40
flags_input.use_spec_augment = 1
flags_input.time_mask_max_size = 25
flags_input.frequency_mask_max_size = 7

NUM_CLASSES = 12
DTYPE = jnp.float32


def cross_entropy_loss(logits, targt):
    # loss function over full time
    # logits = jax.nn.sigmoid(logits)

    targt = jax.nn.one_hot(targt, num_classes=NUM_CLASSES)
    targt = targt * (1.0 - LABEL_SMOOTHING.value) + (
        LABEL_SMOOTHING.value / NUM_CLASSES
    )

    logits = jax.nn.log_softmax(logits, axis=-1)

    return -jnp.mean(jnp.sum(targt * logits, axis=-1))


def compute_metrics(logits, labels):
    # simple MSE loss
    loss = cross_entropy_loss(logits, labels)

    accuracy = jnp.mean(jnp.argmax(logits.sum(0), axis=-1) == labels)

    return {"loss": loss, "accuracy": accuracy}


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(step, optimizer, lr_fn, batch):
    local_batch_size = batch["input_seq"].shape[1]
    init_s = init_state(
        flags_input.dct_num_features,
        local_batch_size,
        HIDDEN_SIZE.value,
        DTYPE,
    )

    # def loss_fn(params):
    #     nn_model_fn = functools.partial(nn_model, params)
    #     final_carry, output_seq = jax.lax.scan(
    #         nn_model_fn, init=init_s, xs=batch["audio"]
    #     )
    #     loss = cross_entropy_loss(output_seq, batch["label"])

    #     # l2 weight decay
    #     params_flat = jnp.hstack(
    #         [x.flatten() for x in jax.tree_util.tree_leaves(params)]
    #     )
    #     return loss + 0.1 * jnp.linalg.norm(params_flat, 2), output_seq

    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss_val, logits), grad = grad_fn(optimizer.target)

    # one hot encode, but we need loss function later on
    targt_one_hot = jax.nn.one_hot(
        batch["target_seq"], num_classes=NUM_CLASSES
    )

    # forward sweep to intialize value nodes
    out_pred, h_pred = forward_sweep(
        batch["input_seq"], optimizer.target, init_s
    )
    # inference to compute error nodes
    e_ys, e_hs = infer(
        optimizer.target,
        batch["input_seq"],
        targt_one_hot,
        out_pred,
        h_pred,
        init_s,
        INFERENCE_STEPS.value,
        INFERENCE_LR.value,
    )
    # compute gradients based on error nodes
    grad = compute_grads(
        optimizer.target, batch["input_seq"], e_ys, e_hs, h_pred
    )

    lr = lr_fn(step)
    optimizer = optimizer.apply_gradient(grad, learning_rate=lr)

    metrics = compute_metrics(out_pred, batch["target_seq"])

    return optimizer, metrics


@jax.jit
def eval_model(params, batch):
    local_batch_size = batch["input_seq"].shape[1]
    nn_model_fn = functools.partial(nn_model, params)
    init_s = init_state(
        flags_input.dct_num_features,
        local_batch_size,
        HIDDEN_SIZE.value,
        DTYPE,
    )

    final_carry, output_seq = jax.lax.scan(
        nn_model_fn, init=init_s, xs=batch["input_seq"]
    )
    metrics = compute_metrics(output_seq, batch["target_seq"])

    return metrics


def main(_):
    summary_writer = tensorboard.SummaryWriter(WORK_DIR.value)
    summary_writer.hparams(
        jax.tree_util.tree_map(lambda x: x.value, flags.FLAGS.__flags)
    )

    # setting up input generator and mfcc processing
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    audio_processor = input_data.AudioProcessor(flags_input)
    time_shift_samples = int(
        (flags_input.time_shift_ms * flags_input.sample_rate) / 1000
    )
    input_audio = tf.keras.layers.Input(
        shape=modes.get_input_data_shape(flags_input, modes.Modes.TRAINING),
        batch_size=BATCH_SIZE.value,
    )
    net = input_audio
    net = speech_features.SpeechFeatures(
        speech_features.SpeechFeatures.get_params(flags_input)
    )(net)
    mfcc_model = tf.keras.Model(input_audio, net)

    # get data set
    rng = jax.random.PRNGKey(SEED.value)

    # initialize parameters
    rng, p_rng = jax.random.split(rng, 2)
    params = init_params(
        p_rng,
        flags_input.dct_num_features,
        NUM_CLASSES,
        1.0,
        HIDDEN_SIZE.value,
    )

    optimizer = optim.Momentum(beta=MOMENTUM.value, nesterov=True).create(
        params
    )
    num_train = audio_processor.set_size("training")
    warmup_steps = int((num_train / BATCH_SIZE.value) * WARMUP_EPOCHS.value)
    learning_rate_fn = create_cosine_learning_rate_schedule(
        LEARNING_RATE.value,
        1.0,
        TRAINING_STEPS.value,
        warmup_length=warmup_steps,
    )

    # Training loop.
    logging.info("Files in: " + WORK_DIR.value)
    logging.info(jax.devices())
    t_loop_start = time.time()
    train_metrics = []
    for step in range(TRAINING_STEPS.value):
        # Do a batch of SGD.
        train_fingerprints, train_ground_truth = audio_processor.get_data(
            BATCH_SIZE.value,
            0,
            flags_input,
            flags_input.background_frequency,
            flags_input.background_volume,
            time_shift_samples,
            "training",
            flags_input.resample,
            flags_input.volume_resample,
            sess,
        )
        mfcc_data = jnp.moveaxis(
            jnp.array(mfcc_model.predict_on_batch(train_fingerprints)),
            (0, 1, 2),
            (1, 0, 2),
        )
        batch = {
            "input_seq": mfcc_data,
            "target_seq": jnp.ones((mfcc_data.shape[0], mfcc_data.shape[1]))
            * jnp.array(train_ground_truth),
            "mask_seq": jnp.ones((mfcc_data.shape[0], mfcc_data.shape[1])),
        }

        optimizer, metrics = train_step(
            step, optimizer, learning_rate_fn, batch
        )
        train_metrics.append(metrics)

        # Periodically report loss
        if (step + 1) % EVALUATION_INTERVAL.value == 0:

            eval_metrics = []
            set_size = audio_processor.set_size("validation")
            set_size = int(set_size / BATCH_SIZE.value) * BATCH_SIZE.value

            for i in range(0, set_size, BATCH_SIZE.value):
                (
                    validation_fingerprints,
                    validation_ground_truth,
                ) = audio_processor.get_data(
                    BATCH_SIZE.value,
                    i,
                    flags_input,
                    0.0,
                    0.0,
                    0,
                    "validation",
                    0.0,
                    0.0,
                    sess,
                )
                mfcc_data = jnp.moveaxis(
                    jnp.array(
                        mfcc_model.predict_on_batch(validation_fingerprints)
                    ),
                    (0, 1, 2),
                    (1, 0, 2),
                )
                batch = {
                    "input_seq": mfcc_data,
                    "target_seq": jnp.ones(
                        (mfcc_data.shape[0], mfcc_data.shape[1])
                    )
                    * jnp.array(validation_ground_truth),
                    "mask_seq": jnp.ones(
                        (mfcc_data.shape[0], mfcc_data.shape[1])
                    ),
                }
                metrics = eval_model(optimizer.target, batch)
                eval_metrics.append(metrics)

            eval_metrics = common_utils.stack_forest(eval_metrics)
            eval_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

            train_metrics = common_utils.stack_forest(train_metrics)
            train_metrics = jax.tree_map(lambda x: x.mean(), train_metrics)

            logging.info(
                "step: %d, train_loss: %.4f, train_accuracy: %.4f, "
                "validation_loss: %.4f, validation_accuracy: %.4f",
                (step + 1),
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
            )
            train_metrics = []

    # testing
    set_size = audio_processor.set_size("testing")
    set_size = int(set_size / BATCH_SIZE.value) * BATCH_SIZE.value
    test_metrics = []

    for i in range(0, set_size, BATCH_SIZE.value):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            BATCH_SIZE.value,
            i,
            flags_input,
            0.0,
            0.0,
            0,
            "testing",
            0.0,
            0.0,
            sess,
        )
        mfcc_data = jnp.moveaxis(
            jnp.array(mfcc_model.predict_on_batch(test_fingerprints)),
            (0, 1, 2),
            (1, 0, 2),
        )
        batch = {
            "input_seq": mfcc_data,
            "target_seq": jnp.ones((mfcc_data.shape[0], mfcc_data.shape[1]))
            * jnp.array(test_ground_truth),
            "mask_seq": jnp.ones((mfcc_data.shape[0], mfcc_data.shape[1])),
        }
        metrics = eval_model(optimizer.target, batch)
        test_metrics.append(metrics)

    test_metrics = common_utils.stack_forest(eval_metrics)
    test_metrics = jax.tree_map(lambda x: x.mean(), eval_metrics)

    logging.info(
        "FINAL LOSS %.4f, FINAL ACCURACY: %.4f on TEST SET",
        test_metrics["loss"],
        test_metrics["accuracy"],
    )


if __name__ == "__main__":
    app.run(main)
