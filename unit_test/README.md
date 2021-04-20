## Predictive Coding Unit Test

Code copied from https://github.com/BerenMillidge/PredictiveCodingBackprop and slightly modified to save gradients after one batch is processed with deterministic initial weights.

To produce gradient files necessary for unit tests run:

```
python rnn.py
```

The copied comes from:
```
@article{millidge2020predictive,
  title={Predictive Coding Approximates Backprop along Arbitrary Computation Graphs},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2006.04182},
  year={2020}
}
```


## Requirements
The code is written in [Pyython 3.x] and uses the following packages:

- [NumPY]
- [PyTorch] version 1.3.1
- [TensorFlow] version 1.x (only for downloading shakespeare dataset)
