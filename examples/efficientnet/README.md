# Quantized EfficientNet Lite

## Performance with Pretrained Weights
|**Model** | **params** | **size FP32** | **size INT8** | **MAdds** | **FP32 acc** | **FP32 acc finetuned** | **INT8 acc** | **INT8 acc finetuned** |
|------|-----|-------|-------|-----|-------|-------|------|------|
|efficientnet-lite0 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite0.tar.gz) | 4.7M | 18.61MB | 4.65MB | 407M |  74.94% | [75.40%](https://tensorboard.dev/experiment/BRj9fv5PR0yAWkD4z0p5FQ/) | 74.22% | |
|efficientnet-lite1 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz) | 5.4M | 21.67MB | 5.42MB | 631M |  76.67% | [76.94%](https://tensorboard.dev/experiment/QRMPo8cVQRqk01JbKZOMjw/) | 76.31% | |
|efficientnet-lite2 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite2.tar.gz) | 6.1M | 24.37MB | 6.09MB | 899M |  77.43% | [77.84%](https://tensorboard.dev/experiment/DZXKGFneSoW8rj5qZZz3LQ/) | 76.91% | |
|efficientnet-lite3 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite3.tar.gz) | 8.2M | 32.79MB | 8.20MB |1.44B | 79.21% | [79.45%](https://tensorboard.dev/experiment/dD3zay4XTYm6ltpNTGocDg/) | 78.87% | |
|efficientnet-lite4 [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz) | 13.0M | 52.03MB | 13.01MB |2.64B | 80.65% | [80.97%](https://tensorboard.dev/experiment/4VwTvygFQ2WFG74GlF8Tqw/) | 80.47% | |

## References

Code based on:
- https://github.com/google/flax/tree/main/examples
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
- https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
