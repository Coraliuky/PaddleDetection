# 1. TTFNet

## Introduction

TTFNet is a training-time-friendly network used for real-time object detection. It improves the slow convergence of CenterNet and proposes to generate training samples using the Gaussian kernel, which effectively eliminates the fuzziness in the anchor-free head. At the same time, its simple and lightweight structure also makes it easy to expand the task.


**Characteristics:**

The structure of TTFNet is simple, requiring only two heads to detect the target position and the size. And the time-consuming post-processing operations are eliminated.
Its training time is short: Based on DarkNet53 backbone network, it only takes 2 hours to obtain a good model with 8 V100 8 cards.

## Model Zoo

| Backbone  | Network type | Number of images per GPU | Learning rate strategy | Inference time(fps) | Box AP |                                     Download                                     |                                                       Configuration file                                                       |
| :-------- | :----------- | :----------------------: | :--------------------: | :-----------------: | :----: | :------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
| DarkNet53 | TTFNet       |            12            |           1x           |        ----         |  33.5  | [Link](https://paddledet.bj.bcebos.com/models/ttfnet_darknet53_1x_coco.pdparams) | [Configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ttfnet/ttfnet_darknet53_1x_coco.yml) |





# 2. PAFNet

## Introduction

PAFNet (Paddle Anchor-Free Network) is an optimized version of TTFNet in PaddleDetection. Its accuracy is the most advanced among the anchor-free counterparts, and meanwhile it has a mobile lightweight version-- PAFNet-Lite.

PAFNet series have optimized the TTFNet model on:

- [CutMix](https://arxiv.org/abs/1905.04899)
- Better backbone network: ResNet50vd-DCN
- Larger training batch size: 8 GPUs, batch size=18 for each
- Synchronized Batch Normalization
- [Deformable Convolution](https://arxiv.org/abs/1703.06211)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- Better pretrained model


## Model Library

| Backbone   | Net type | Number of images per GPU | Learning rate strategy | Inference time(fps) | Box AP |                                Download                                 |                                                  Configuration file                                                   |
| :--------- | :------- | :----------------------: | :--------------------: | :-----------------: | :----: | :---------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| ResNet50vd | PAFNet   |            18            |          10x           |        ----         |  39.8  | [Link](https://paddledet.bj.bcebos.com/models/pafnet_10x_coco.pdparams) | [Configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ttfnet/pafnet_10x_coco.yml) |



### PAFNet-Lite

| Backbone    | Net type    | Number of images per GPU | Learning rate strategy | Box AP | Kirin 990 delay（ms） | volume（M） |                                         Download                                          |                                                           Configuration file                                                            |
| :---------- | :---------- | :----------------------: | :--------------------: | :----: | :-------------------: | :---------: | :---------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| MobileNetv3 | PAFNet-Lite |            12            |          20x           |  23.9  |         26.00         |     14      | [Link](https://paddledet.bj.bcebos.com/models/pafnet_lite_mobilenet_v3_20x_coco.pdparams) | [Configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ttfnet/pafnet_lite_mobilenet_v3_20x_coco.yml) |

**Note:** Due to the overall upgrade of the dynamic graph framework, the weight model published by PaddleDetection of PAFNet needs to be evaluated with a --bias field, for example,

```bash
# Use weights published by Paddle Detection
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/pafnet_10x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/pafnet_10x_coco.pdparams --bias
```

## Citations
```
@article{liu2019training,
  title   = {Training-Time-Friendly Network for Real-Time Object Detection},
  author  = {Zili Liu, Tu Zheng, Guodong Xu, Zheng Yang, Haifeng Liu, Deng Cai},
  journal = {arXiv preprint arXiv:1909.00700},
  year    = {2019}
}
```
