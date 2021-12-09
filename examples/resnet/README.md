# Quantized ResNet

## Performance with Pretrained Weights
|**Model** | **Dataset** | **Config** |**#Params** | **Size** | **Accuracy** |
|------|-------|-------|-----|-------|------|
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip) | ImageNet | [FP32](configs/resnet18_fp32.py) | 11M | 18.61MB |  [70.75%](https://tensorboard.dev/experiment/2ClIM4T0TjOEcekcLFPXbQ) |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| ImageNet | [INT8 PQT](configs/resnet18_w8a8.py)| 11M | 4.65MB | |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| ImageNet | [MIXED QAT](configs/resnet18_mixed.py)| | | |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip) | CIFAR10 | [FP32](configs/resnet20_fp_cifar10.py) | | | |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| CIFAR10 | [INT8 PQT](configs/resnet20_w8a8_cifar10.py)| | | |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| CIFAR10 | [MIXED QAT](configs/resnet20_mixed_cifar10.py)| | | |

## References

Code based on:
- https://github.com/google/flax/tree/main/examples
- https://github.com/sony/ai-research-code/tree/master/mixed-precision-dnns
