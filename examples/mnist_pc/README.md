## MNIST Task

Trains a simple network on the MNIST task with predictive coding based on [An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity](https://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00949).

You can run this code and even modify it directly in Google Colab, no
installation required:

https://colab.research.google.com/github/Intelligent-Microsystems-Lab/trainingSNNs/blob/main/examples/mnist_pc/mnist_pc.ipynb

### Example output


```
I0308 13:34:21.753504 4486004224 train.py:242] epoch: 1, train loss: 0.1980, train acc: 0.7667, eval loss: 0.1652, eval acc: 0.8774 
I0308 13:36:29.745707 4486004224 train.py:242] epoch: 2, train loss: 0.1543, train acc: 0.8915, eval loss: 0.1456, eval acc: 0.9035 
I0308 13:38:33.516435 4486004224 train.py:242] epoch: 3, train loss: 0.1424, train acc: 0.9121, eval loss: 0.1324, eval acc: 0.9244 
I0308 13:40:32.653883 4486004224 train.py:242] epoch: 4, train loss: 0.1356, train acc: 0.9216, eval loss: 0.1260, eval acc: 0.9336 
I0308 13:42:40.477018 4486004224 train.py:242] epoch: 5, train loss: 0.1309, train acc: 0.9286, eval loss: 0.1351, eval acc: 0.9336 
I0308 13:44:44.877048 4486004224 train.py:242] epoch: 6, train loss: 0.1277, train acc: 0.9311, eval loss: 0.1215, eval acc: 0.9401 
I0308 13:46:47.323173 4486004224 train.py:242] epoch: 7, train loss: 0.1256, train acc: 0.9347, eval loss: 0.1321, eval acc: 0.9242 
I0308 13:48:53.092221 4486004224 train.py:242] epoch: 8, train loss: 0.1239, train acc: 0.9352, eval loss: 0.1155, eval acc: 0.9377 
I0308 13:51:03.806893 4486004224 train.py:242] epoch: 9, train loss: 0.1221, train acc: 0.9368, eval loss: 0.1236, eval acc: 0.9356 
I0308 13:53:10.215139 4486004224 train.py:242] epoch: 10, train loss: 0.1202, train acc: 0.9398, eval loss: 0.1189, eval acc: 0.9382 
```

Note: resutlts were obtained with different variace at output.

### How to run

`python main.py --workdir=/tmp/mnist --config=configs/default.py`

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
