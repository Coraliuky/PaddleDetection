# Deformable DETR

## Introduction


Deformable DETR is an object detection model based on DETR. We have reproduced the model of the paper.


## Model Zoo

| Backbone | Model | Number of images/GPU  | Inference time (fps) | Box AP | Config | Download |
|:------:|:--------:|:--------:|:--------------:|:------:|:------:|:--------:|
| R-50 | Deformable DETR  | 2 | --- | 44.1 | [Config](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/deformable_detr/deformable_detr_r50_1x_coco.yml) | [Model](https://paddledet.bj.bcebos.com/models/deformable_detr_r50_1x_coco.pdparams) |

**Note:**

- Deformable DETR is trained on COCO train2017 dataset and evaluated on val2017 in `mAP(IoU=0.5:0.95)`.
- Deformable DETR uses 8 GPUs to train 50 epochs.

GPU multi-card training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/deformable_detr/deformable_detr_r50_1x_coco.yml --fleet -o find_unused_parameters=True
```

## Citations
```
@inproceedings{
zhu2021deformable,
title={Deformable DETR: Deformable Transformers for End-to-End Object Detection},
author={Xizhou Zhu and Weijie Su and Lewei Lu and Bin Li and Xiaogang Wang and Jifeng Dai},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=gZ9hCDWe6ke}
}
```
