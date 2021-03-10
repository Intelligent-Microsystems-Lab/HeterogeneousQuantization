"""Tests for copy task bptt train."""

import tempfile

from absl.testing import absltest
import train
from configs import default  # type: ignore
import jax
from jax import numpy as jnp


# Require JAX omnistaging mode.
jax.config.enable_omnistaging()


class MnistLibTest(absltest.TestCase):
    """Test cases for mnist_lib."""

    def test_net(self):
        """Tests CNN module used as the trainable model."""
        rng = jax.random.PRNGKey(0)
        inputs = jnp.ones((10, 2), jnp.float32)
        output, variables = train.Net().init_with_output(rng, inputs)

        self.assertEqual((10, 1), output.shape)

        # TODO(mohitreddy): Consider creating a testing module which
        # gives a parameters overview including number of parameters.
        self.assertLen(variables["params"], 2)

    def test_train_and_evaluate(self):
        """Tests training and evaluation code by running a single step."""
        # Create a temporary directory where tensorboard metrics are written.
        workdir = tempfile.mkdtemp()

        # Define training configuration.
        config = default.get_config()
        config.num_epochs = 1
        config.batch_size = 8

        train.train_and_evaluate(config=config, workdir=workdir)


if __name__ == "__main__":
    absltest.main()
