"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # random seed
    config.random_seed = 193012823
    # learning rate
    config.learning_rate = 0.001

    # Dimensionality of each vector to copy
    config.num_bits = 4
    # Batch size for training
    config.batch_size = 128
    # Lower limit on number of vectors in the observation pattern to copy
    config.min_length = 1
    # Upper limit on number of vectors in the observation pattern to copy
    config.max_length = 2
    # Lower limit on number of copy repeats.
    config.min_repeats = 1
    # Upper limit on number of copy repeats.
    config.max_repeats = 2

    config.hidden_units = 256

    config.num_steps = 5000

    config.report_interval = 100

    return config
