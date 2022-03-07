# Generalized Focal Loss Model(GFL)

## Introduction

We have reproduced the object detection results in the papers [Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://arxiv.org/abs/2006.04388) and [Generalized Focal Loss V2](https://arxiv.org/pdf/2011.12885.pdf). A better pre-trained model and the ResNet-vd structure are used to improve mAP.

## Model Zoo

| Backbone        | Model      | Batch-size/GPU | lr schedule |FPS | Box AP |                           Download                          | Config |
| :-------------- | :------------- | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| ResNet50    | GFL           |    2    |   1x      |     ----     |  41.0  | [Model](https://paddledet.bj.bcebos.com/models/gfl_r50_fpn_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r50_fpn_1x_coco.log) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/gfl/gfl_r50_fpn_1x_coco.yml) |
| ResNet101-vd   | GFL           |    2    |   2x      |     ----     |  46.8  | [Model](https://paddledet.bj.bcebos.com/models/gfl_r101vd_fpn_mstrain_2x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r101vd_fpn_mstrain_2x_coco.log) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/gfl/gfl_r101vd_fpn_mstrain_2x_coco.yml) |
| ResNet34-vd    | GFL           |    2    |   1x      |     ----     |  40.8  | [Model](https://paddledet.bj.bcebos.com/models/gfl_r34vd_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r34vd_1x_coco.log) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/gfl/gfl_r34vd_1x_coco.yml) |
| ResNet18-vd   | GFL           |    2    |   1x      |     ----     |  36.6  | [Model](https://paddledet.bj.bcebos.com/models/gfl_r18vd_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gfl_r18vd_1x_coco.log) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/gfl/gfl_r18vd_1x_coco.yml) |
| ResNet50    | GFLv2       |    2    |   1x      |     ----     |  41.2  | [Model](https://paddledet.bj.bcebos.com/models/gflv2_r50_fpn_1x_coco.pdparams) &#124; [log](https://paddledet.bj.bcebos.com/logs/train_gflv2_r50_fpn_1x_coco.log) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/gfl/gflv2_r50_fpn_1x_coco.yml) |


**Note:**

- GFL is trained on COCO train2017 dataset with 8 GPUs and evaluated on val2017 with `mAP(IoU=0.5:0.95)`.

## Citations
```
@article{li2020generalized,
  title={Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}

@article{li2020gflv2,
  title={Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection},
  author={Li, Xiang and Wang, Wenhai and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2011.12885},
  year={2020}
}

```
