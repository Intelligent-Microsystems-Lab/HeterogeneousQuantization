## MNIST Task

Trains a simple network on the MNIST task with predictive coding.

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/Intelligent-Microsystems-Lab/trainingSNNs/blob/main/examples/mnist_pc/mnist.ipynb

### Example output


```

```

### How to run

`python main.py --workdir=/tmp/xor --config=configs/default.py`

#### Overriding Hyperparameter configurations

MNIST example allows specifying a hyperparameter configuration by the means of
setting `--config` flag. Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py \
--workdir=/tmp/mnist --config=configs/default.py \
--config.inference_iterations=10 --config.num_epochs=5
```
