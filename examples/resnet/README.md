# Quantized ResNet

Note: results obtained with commit 845652fd70e3ee025d9550f6e71cb7ea207bb4f5

## Performance
|**Model** | **Dataset** | **Config** |**#Params** | **Size** | **Accuracy** |
|------|-------|-------|-----|-------:|------|
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip) | ImageNet | [FP32](configs/resnet18_fp32.py) | 11M | 18.61MB |  [70.75%](https://tensorboard.dev/experiment/2ClIM4T0TjOEcekcLFPXbQ) |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| ImageNet | [INT8 PQT](configs/resnet18_int8.py)| 11M | 11.68MB | 70.62% |
|ResNet18 [ckpt]()| ImageNet | [INT8 QAT](configs/resnet18_int4.py)| 11M | 11.68MB | |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip)| ImageNet | [INT4 PQT](configs/resnet18_int4.py)| 11M | 5.84MB | 9.01% |
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5bo9b8twe87i96a59dsz0wp6wk6unfyr.zip)| ImageNet | [INT4 QAT](configs/resnet18_int4.py)| 11M | 5.84MB | [69.94%](https://tensorboard.dev/experiment/ElAWUvdARaKibpmM1e4WOA/)|
|ResNet18 [ckpt]()| ImageNet | [MIXED QAT](configs/resnet18_mixed.py)| 11M | 5.40MB | |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip) | CIFAR10 | [FP32](configs/resnet20_fp_cifar10.py) | 0.27M | 1072KB | [95.19%](https://tensorboard.dev/experiment/apemnH67RXeI5VvrfWl7jg/) |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip)| CIFAR10 | [INT8 PQT](configs/resnet20_int8_cifar10.py)| 0.27M | 268KB | 14.62% |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/3a6aw491lp5u6bv9dvz65b7o7tuh0osr.zip)| CIFAR10 | [INT8 QAT](configs/resnet20_int8_cifar10.py)| 0.27M | 268KB | [93.83%](https://tensorboard.dev/experiment/nfqozT7BQnyE08TJf9EQ2w/) |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/z1pxy1b5poz8cdarg4wgm2jp4esg0mji.zip)| CIFAR10 | [INT4 PQT](configs/resnet20_int4_cifar10.py)| 0.27M | 134KB | 13.16% |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/gcyf44n4vdzhjsgukga07dz804cspczb.zip)| CIFAR10 | [INT4 QAT](configs/resnet20_int4_cifar10.py)| 0.27M | 134KB | [93.43%](https://tensorboard.dev/experiment/nKLF9KgWTjejSnhmvkbcXg/) |
|ResNet20 [ckpt](https://notredame.box.com/shared/static/2ap5ckl6i77eoga07313gnwrl4vbptku.zip)| CIFAR10 | [MIXED QAT](configs/resnet20_mixed_cifar10.py)| 0.27M | 70KB | [93.51%](https://tensorboard.dev/experiment/HqWb4LDLQZeZDCvpgkN5Uw/) |

## References

Code based on:
- https://github.com/google/flax/tree/main/examples
- https://github.com/sony/ai-research-code/tree/master/mixed-precision-dnns
