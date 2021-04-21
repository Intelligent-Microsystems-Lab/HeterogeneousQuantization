# copied from https://github.com/BerenMillidge/PredictiveCodingBackprop

import tensorflow as tf
import numpy as np


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def get_lstm_dataset(seq_length, batch_size, buffer_size=10000):
    path_to_file = tf.keras.utils.get_file(
        "shakespeare.txt",
        "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
    )
    text = open(path_to_file, "rb").read().decode(encoding="utf-8")
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(buffer_size).batch(
        batch_size, drop_remainder=True
    )
    dataset = list(iter(dataset))
    # get dataset in right format
    vocab_size = len(vocab)
    return dataset, vocab_size, char2idx, idx2char
