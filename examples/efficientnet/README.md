# Quantized EfficientNet Lite

## Performance with Pretrained Weights
|**Model** | **params** | **MAdds** | **FP32 accuracy** | **FP32 CPU  latency** | **FP32 GPU latency** |**INT8 accuracy** | **INT8 CPU latency**  | **INT8 TPU latency**|
|------|-----|-------|-------|-------|-------|-------|-------|-------|-------|
|efficientnet-lite0 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz) | 4.7M | 407M |  66.92% |  ?? | 10.4258±7.0528ms | ??  |  ?? | ?? |
|efficientnet-lite1 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz) | 5.4M | 631M |  70.19% |  ?? | 13.1659±6.5807ms | ??  |  ?? | ?? |
|efficientnet-lite2 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite2.tar.gz) | 6.1M | 899M |  68.85% |  ?? | 13.9165±6.3518ms | ?? | ?? | ?? |
|efficientnet-lite3 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz) | 8.2M | 1.44B |  ?? |  ?? | ?? | ??  | ?? | ?? |
|efficientnet-lite4 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz) |13.0M | 2.64B |  ?? |  ?? | ?? | ??  | ?? | ?? |



## References

- https://github.com/google/flax/tree/main/examples
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
