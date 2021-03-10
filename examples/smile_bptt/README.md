## Smile Spatio-Temporal Pattern Retention with BPTT

based on https://github.com/bamsumit/slayerPytorch and https://proceedings.neurips.cc/paper/2018/file/82f2b308c3b01637c607ce05f52a2fed-Paper.pdf

Trains a SNN to retain a spatio-temporal pattern with backpropagation through time (BPTT).

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/Intelligent-Microsystems-Lab/trainingSNNs/blob/main/examples/???

### Example output

```
????
```

### How to run

`python main.py --workdir=/tmp/xor --config=configs/default.py`

#### Overriding Hyperparameter configurations

Smile example allows specifying a hyperparameter configuration by the means of
setting `--config` flag. Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py \
--workdir=/tmp/mnist --config=configs/default.py \
--config.num_epochs=5
```
