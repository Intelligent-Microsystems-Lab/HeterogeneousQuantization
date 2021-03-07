"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # random seed
    config.random_seed = 193012823
    # learning rate
    config.learning_rate = 0.2
    # iterations for inference convergence
    config.inference_iterations = 100
    # number of training epochs
    config.num_epochs = 500
    # euler integration constant
    config.beta = 0.2

    # plot every x epochs
    config.plotevery = 50

    return config
