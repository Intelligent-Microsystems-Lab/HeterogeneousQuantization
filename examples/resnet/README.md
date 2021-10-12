# Quantized ResNet

## Performance with Pretrained Weights
|**Model** | **params** | **size FP32** | **size INT8** | **FP32 acc** | **INT8 acc** | 
|------|-------|-------|-----|-------|------|
|ResNet18 [ckpt](https://notredame.box.com/shared/static/5m485mqpskw5lwop1z3yfi4wsvvdjsx3.zip) | 11M | 18.61MB | 4.65MB |  [70.65%](https://tensorboard.dev/experiment/83y0Ro6lTyu2JzP16bbm7w) | 70.34% |


## References

Code based on:
- https://github.com/google/flax/tree/main/examples
