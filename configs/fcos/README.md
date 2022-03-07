# FCOS for Object Detection

## Introduction

FCOS (Fully Convolutional One-Stage Object Detection) is a fast anchor-free object detection framework with strong performance. We have reproduced the model of the paper and optimized the accuracy of the FCOS.

**Highlight:**

- Training Time: Training the model of `fcos_r50_fpn_1x` on Tesla v100 with 8 GPUs only takes 8.5 hours.

## Model Zoo

| Backbone        | Model      | Number of images per GPU | lr schedule |FPS | Box AP |                           Download                          | Config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50-FPN    | FCOS           |    2    |   1x      |     ----     |  39.6  | [Download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/fcos/fcos_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS+DCN       |    2    |   1x      |     ----     |  44.3  | [Download](https://paddledet.bj.bcebos.com/models/fcos_dcn_r50_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/fcos/fcos_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-FPN    | FCOS+multiscale_train    |    2    |   2x      |     ----     |  41.8  | [Download](https://paddledet.bj.bcebos.com/models/fcos_r50_fpn_multiscale_2x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/fcos/fcos_r50_fpn_multiscale_2x_coco.yml) |

**Note:**

- FCOS is trained on COCO train2017 dataset and evaluated on val2017 in `mAP(IoU=0.5:0.95)`.

## Citations
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```
