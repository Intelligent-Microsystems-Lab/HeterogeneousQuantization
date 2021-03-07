## XOR Task

Trains a simple network on the XOR task with predictive coding.

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/Intelligent-Microsystems-Lab/trainingSNNs/blob/main/examples/xor_pc/XOR.ipynb


### Example output


```
I0307 01:15:51.832773 4592487936 train.py:130] epoch: 0, train rmse: -.----, test rmse: 0.5773 
I0307 01:16:05.905062 4592487936 train.py:145] epoch: 50, train rmse: 0.4478, test rmse: 0.3135 
I0307 01:16:05.947964 4592487936 train.py:145] epoch: 100, train rmse: 0.0028, test rmse: 0.0017 
I0307 01:16:05.993340 4592487936 train.py:145] epoch: 150, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.038058 4592487936 train.py:145] epoch: 200, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.081298 4592487936 train.py:145] epoch: 250, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.125264 4592487936 train.py:145] epoch: 300, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.167360 4592487936 train.py:145] epoch: 350, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.209349 4592487936 train.py:145] epoch: 400, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.255126 4592487936 train.py:145] epoch: 450, train rmse: 0.0000, test rmse: 0.0000 
I0307 01:16:06.300794 4592487936 train.py:145] epoch: 500, train rmse: 0.0000, test rmse: 0.0000 
```

### How to run

`python main.py --workdir=/tmp/xor --config=configs/default.py`

#### Overriding Hyperparameter configurations

XOR example allows specifying a hyperparameter configuration by the means of
setting `--config` flag. Configuration flag is defined using
[config_flags](https://github.com/google/ml_collections/tree/master#config-flags).
`config_flags` allows overriding configuration fields. This can be done as
follows:

```shell
python main.py \
--workdir=/tmp/mnist --config=configs/default.py \
--config.inference_iterations=10 --config.num_epochs=5
```
