"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # random seed
    config.random_seed = 193012823
    # learning rate
    config.learning_rate = 0.001
    # iterations for inference convergence
    config.inference_iterations = 100
    # number of training epochs
    config.num_epochs = 10
    # euler integration constant
    config.beta = 0.2

    # data directory
    config.data_dir = "../../../data"
    # batch size
    config.batch_size = 20
    # activation function
    config.act_fn = "sigmoid"
    # last layer fwd variance
    config.sigma_0 = 100

    return config
