# Quantized EfficientNet Lite

## Performance with Pretrained Weights
|**Model** | **params** | **MAdds** | **FP32 accuracy** |
|------|-----|-------|-------|
|efficientnet-lite0 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz) | 4.7M | 407M |  74.94% |
|efficientnet-lite1 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz) | 5.4M | 631M |  76.67% |
|efficientnet-lite2 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite2.tar.gz) | 6.1M | 899M |  77.43% |
|efficientnet-lite3 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz) | 8.2M | 1.44B | 79.21% |
|efficientnet-lite4 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz) |13.0M | 2.64B | 80.65% |

## References

Code based on:
- https://github.com/google/flax/tree/main/examples
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
