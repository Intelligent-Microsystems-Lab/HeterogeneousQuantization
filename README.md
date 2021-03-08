# Training SNNs


## Running on CRC GPU

```
module load python cuda/10.2
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/afs/crc.nd.edu/x86_64_linux/c/cuda/10.2"
```

## References

Whittington JCR, Bogacz R (2017) An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity. Neural Comput 29:1229-1262