English | [简体中文](README_cn.md)
# Detection Models for Specific Scenarios

We provide some models implemented by PaddlePaddle to detect objects in specific scenarios, so users can download the models and use them.

| Task                 | Algorithm | Box AP | Download                                                                                | Configs |
|:---------------------|:---------:|:------:| :-------------------------------------------------------------------------------------: |:------:|
| Pedestrian Detection |  YOLOv3  |  51.8  | [Model](https://paddledet.bj.bcebos.com/models/pedestrian_yolov3_darknet.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/pedestrian/pedestrian_yolov3_darknet.yml) |

## Pedestrian Detection

One application scenario of pedetestrian detection is intelligent monitoring, where photos of pedetestrians are taken by surveillance cameras in public areas, then detection are conducted on the captured photos.

### 1. Network

Backbone: YOLOv3 for the Darknet53 approach

### 2. Configuration for Training

PaddleDetection provides users with a configuration file [yolov3_darknet53_270e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml) to train YOLOv3 on the COCO dataset. By comparison, we have modified some parameters of the pedestrian detection training:

* num_classes: 1
* dataset_dir: dataset/pedestrian

### 3. Accuracy

The accuracy of the model trained and evaluted on private data is:

AP at IoU=.50:.05:.95 is 0.518.

AP at IoU=.50 is 0.792.

### 4. Inference

Users can employ the trained model to conduct the inference:

```
export CUDA_VISIBLE_DEVICES=0
python -u tools/infer.py -c configs/pedestrian/pedestrian_yolov3_darknet.yml \
                         -o weights=https://paddledet.bj.bcebos.com/models/pedestrian_yolov3_darknet.pdparams \
                         --infer_dir configs/pedestrian/demo \
                         --draw_threshold 0.3 \
                         --output_dir configs/pedestrian/demo/output
```

Some inference results are visualized below:

![](../../docs/images/PedestrianDetection_001.png)

![](../../docs/images/PedestrianDetection_004.png)
