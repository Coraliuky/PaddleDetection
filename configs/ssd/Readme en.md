# SSD: Single Shot MultiBox Detector

## Model Zoo

### SSD on Pascal VOC

| Backbone network  | Net type |  Number of images per GPU | Learning rate strategy| Inference time (fps) | Box AP  | Download  |  Config  |
| :---------------: | :------: | :-----------------------: | :-------------------: | :-------------------:| :-----: | :-------: | :------: |
|  VGG| SSD| 8 | 240e | ---- | 77.8 | [Link](https://paddledet.bj.bcebos.com/models/ssd_vgg16_300_240e_voc.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ssd/ssd_vgg16_300_240e_voc.yml)|
| MobileNet v1 | SSD | 32 | 120e | ---- | 73.8 | [Link](https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml) |

**Note:** 
SSD-VGG uses 4 GPUs to train 240 epochs (Batch size= 32). SSD-MobileNetv1 uses 2 GPUs to train 12o epochs (Batch size=64).

## Citations
```
@article{Liu_2016,
   title={SSD: Single Shot MultiBox Detector},
   journal={ECCV},
   author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
   year={2016},
   }
   ```
