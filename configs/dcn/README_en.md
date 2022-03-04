[简体中文](README.md) | English
### Deformable ConvNets v2

| Backbone |  Net type | Convolution | Number of images per GPU | Learning rate strategy | Inference time (fps) | Box AP | Mask AP | Download | Config|
| :-------:| :--------:| :----------:| :----------------------: | :--------------------: | :------------------: | :----: | :-----: | :------: | :---: |
| ResNet50-FPN  | Faster | c3-c5 | 1 | 1x | - | 42.1 | - | [Model](https://paddledet.bj.bcebos.com/models/faster_rcnn_dcn_r50_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/faster_rcnn_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-vd-FPN | Faster | c3-c5 | 1 | 1x | - | 42.7 | - | [Model](https://paddledet.bj.bcebos.com/models/faster_rcnn_dcn_r50_vd_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/faster_rcnn_dcn_r50_vd_fpn_1x_coco.yml) |
| ResNet50-vd-FPN | Faster | c3-c5 | 1 | 2x | - | 43.7 | - | [Model](https://paddledet.bj.bcebos.com/models/faster_rcnn_dcn_r50_vd_fpn_2x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/faster_rcnn_dcn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-vd-FPN     | Faster         | c3-c5   |    1      |   1x    |    -     |  45.1  |    -    | [Model](https://paddledet.bj.bcebos.com/models/faster_rcnn_dcn_r101_vd_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/faster_rcnn_dcn_r101_vd_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Faster         | c3-c5   |    1      |   1x    |    -     |  46.5  |    -    | [Model](https://paddledet.bj.bcebos.com/models/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco.pdparams) |[Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/faster_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco.yml) |
| ResNet50-FPN         | Mask           | c3-c5   |    1      |   1x    |    -     |  42.7  |   38.4   | [Model](https://paddledet.bj.bcebos.com/models/mask_rcnn_dcn_r50_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/mask_rcnn_dcn_r50_fpn_1x_coco.yml) |
| ResNet50-vd-FPN      | Mask           | c3-c5   |    1      |   2x    |    -     |  44.6  |  39.8   | [Model](https://paddledet.bj.bcebos.com/models/mask_rcnn_dcn_r50_vd_fpn_2x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/mask_rcnn_dcn_r50_vd_fpn_2x_coco.yml) |
| ResNet101-vd-FPN     | Mask           | c3-c5   |    1      |   1x    |    -     |  45.6 |  40.6  | [Model](https://paddledet.bj.bcebos.com/models/mask_rcnn_dcn_r101_vd_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/mask_rcnn_dcn_r101_vd_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Mask           | c3-c5   |    1      |   1x    |     -    |  47.3 |  42.0  | [Model](https://paddledet.bj.bcebos.com/models/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/mask_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco.yml) |
| ResNet50-FPN         | Cascade Faster | c3-c5 |    1   |   1x    |  -   |  42.1  |    -    | [Model](https://paddledet.bj.bcebos.com/models/cascade_rcnn_dcn_r50_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/cascade_rcnn_dcn_r50_fpn_1x_coco.yml) |
| ResNeXt101-vd-FPN    | Cascade Faster | c3-c5 |    1   |   1x    |    -   | 48.8 |  -  | [Model](https://paddledet.bj.bcebos.com/models/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dcn/cascade_rcnn_dcn_x101_vd_64x4d_fpn_1x_coco.yml) |


**Note:**

- Deformable ConvNets v2 (dcn_v2) refer to the paper [Deformable ConvNets v2](https://arxiv.org/abs/1811.11168).
- `c3-c5` means adding `dcn` in resnet stage 3-5.

## Citations
```
@inproceedings{dai2017deformable,
  title={Deformable Convolutional Networks},
  author={Dai, Jifeng and Qi, Haozhi and Xiong, Yuwen and Li, Yi and Zhang, Guodong and Hu, Han and Wei, Yichen},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  year={2017}
}
@article{zhu2018deformable,
  title={Deformable ConvNets v2: More Deformable, Better Results},
  author={Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  journal={arXiv preprint arXiv:1811.11168},
  year={2018}
}
```
