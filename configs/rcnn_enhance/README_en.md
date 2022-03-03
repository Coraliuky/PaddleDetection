## Practical Server Side Detection

### Introduction

* In recent years, the object detection of images has been widely concerned by academia and the industrial sector. ResNet50vd pretraining model is obtained by using the SSLD (Simple Semi-supervised Label Knowledge Distillation) strategy of [PaddleClas](https://github.com/PaddlePaddle/PaddleClas) whose Top1 Acc on ImageNet1k validation set is 82.39%. Combining the model and the rich operators of PaddleDetection, PaddlePaddle provides a PSS-DET(Practical Server-Side Detection) scheme. Based on COCO2017 object detection dataset, its inference speed on V100 gpu is 61FPS, and its COCO mAP can reach 41.2%.


### Model library

| Backbone              | Network type | Number of images per GPU | Learning rate strategy | Inference time(fps) | Box AP | Mask AP |                                      Download                                       |                                                           Configuration File                                                            |
| :-------------------- | :----------: | :----------------------: | :--------------------: | :-----------------: | :----: | :-----: | :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| ResNet50-vd-FPN-Dcnv2 |    Faster    |            2             |           3x           |       61.425        |  41.5  |    -    | [Link](https://paddledet.bj.bcebos.com/models/faster_rcnn_enhance_3x_coco.pdparams) | [Configuration File](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/rcnn_enhance/faster_rcnn_enhance_3x_coco.yml) |
