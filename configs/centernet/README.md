English | [简体中文](README_cn.md)

# CenterNet (CenterNet: Objects as Points)

## Table of Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Citations](#Citations)

## Introduction

[CenterNet](http://arxiv.org/abs/1904.07850) is an anchor-free detector, which detects an object as a single point -- the center point of its bounding box.  The Keypoint-based detector locates the center point of a box and regresses other attributes of an object. More importantly, CenterNet is an end-to-end trainable network based on center points, more efficient and accurate than its anchor-based counterparts.

## Model Zoo

### Results of CenterNet on COCO-val 2017

| Backbone       | Input shape | mAP   |    FPS    | Download | Config |
| :--------------| :------- |  :----: | :------: | :----: |:-----: |
| DLA-34(paper)  | 512x512 |  37.4  |     -   |    -   |   -    |
| DLA-34         | 512x512 |  37.6  |     -   | [Model](https://bj.bcebos.com/v1/paddledet/models/centernet_dla34_140e_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/centernet/centernet_dla34_140e_coco.yml) |
| ResNet50 + DLAUp  | 512x512 |  38.9  |     -   | [Model](https://bj.bcebos.com/v1/paddledet/models/centernet_r50_140e_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/centernet/centernet_r50_140e_coco.yml) |

## Citations
```
@article{zhou2019objects,
  title={Objects as points},
  author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
  journal={arXiv preprint arXiv:1904.07850},
  year={2019}
}
```
