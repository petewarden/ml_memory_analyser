# ml_memory_analyser
Utilities to explore runtime memory usage of machine learning models

## Installation

```bash
pip install flatbuffers==1.12.0
```

| Model        | RAM required    |
| ------------ | --------------- |
| MobileNet v1 | 1,204,224 bytes |
| MobileNet v2 | 1,505,280 bytes |
| MobileNet v3 | 1,003,520 bytes |
| EfficientNet Lite 0 | 1,505,280 bytes |
| EfficientNet Lite 4 | 4,050,000 bytes |
| ResNet v2 101 | 4,320,000 bytes |
| Whisper Tiny En | 28,152,000 bytes |



## Models

http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz
https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz

https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-small_224_1.0_uint8.tgz
https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_uint8.tgz

https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite