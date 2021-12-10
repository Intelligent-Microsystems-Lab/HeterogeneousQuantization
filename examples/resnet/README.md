# Quantized ResNet

## Performance
|**Model** | **Dataset** | **Config** |**#Params** | **Size** | **Accuracy** |
|------|-------|-------|-----|-------:|------|
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip) | ImageNet | [FP32](configs/resnet18_fp32.py) | 11M | 18.61MB |  [70.75%](https://tensorboard.dev/experiment/2ClIM4T0TjOEcekcLFPXbQ) |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| ImageNet | [INT8 PQT](configs/resnet18_w8a8.py)| 11M | 11.68MB | |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| ImageNet | [INT8 QAT](configs/resnet18_w8a8.py)| 11M | 11.68MB | |
|ResNet18 [ckpt]()| ImageNet | [MIXED QAT](configs/resnet18_mixed.py)| 11M | 5.40MB | |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip) | CIFAR10 | [FP32](configs/resnet20_fp_cifar10.py) | 0.27M | 1072KB | [95.19%](https://tensorboard.dev/experiment/apemnH67RXeI5VvrfWl7jg/) |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip)| CIFAR10 | [INT8 PQT](configs/resnet20_int8_cifar10.py)| 0.27M | 268KB | 14.62% |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/3a6aw491lp5u6bv9dvz65b7o7tuh0osr.zip)| CIFAR10 | [INT8 QAT](configs/resnet20_int8_cifar10.py)| 0.27M | 268KB | [93.83%](https://tensorboard.dev/experiment/nfqozT7BQnyE08TJf9EQ2w/) |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip)| CIFAR10 | [INT4 PQT](configs/resnet20_int4_cifar10.py)| 0.27M | 134KB | 13.16% |
|ResNet20 [ckpt]()| CIFAR10 | [INT4 QAT](configs/resnet20_int4_cifar10.py)| 0.27M | 134KB | |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/2ap5ckl6i77eoga07313gnwrl4vbptku.zip)| CIFAR10 | [MIXED QAT](configs/resnet20_mixed_cifar10.py)| 0.27M | 70KB | [92.47%](https://tensorboard.dev/experiment/HVNvbxQvRumHvEl3JRseLQ) |

## References

Code based on:
- https://github.com/google/flax/tree/main/examples
- https://github.com/sony/ai-research-code/tree/master/mixed-precision-dnns
