English | [简体中文](README_cn.md)
# PaddleDetection Applied for Specific Scenarios

We provide some models implemented by PaddlePaddle to detect objects in some scenarios, and users can download them.

| Task                 | Algorithm | Box AP | Download                                                                                | Configs |
|:--------------------:|:---------:|:------:| :-------------------------------------------------------------------------------------: |:------:|
| Vehicle Detection    |  YOLOv3  |  54.5  | [Link](https://paddledet.bj.bcebos.com/models/vehicle_yolov3_darknet.pdparams) | [Config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/vehicle/vehicle_yolov3_darknet.yml) |

## Vehicle Detection

One of major applications of vehichle detection is traffic monitoring. In this scenario, images of vehicles to-be-detected are mostly captured by cameras mounted on top of traffic light posts.

### 1. Network

The network for detecting vehicles is YOLOv3, whose backbone is Dacknet53.

### 2. Configuration for Training

PaddleDetection provides users with a configuration file [yolov3_darknet53_270e_coco.yml](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml) to train YOLOv3 on the COCO dataset. In this case, we have modified some parameters to train models of vehicle detection:

* num_classes: 6
* anchors: [[8, 9], [10, 23], [19, 15], [23, 33], [40, 25], [54, 50], [101, 80], [139, 145], [253, 224]]
* nms/nms_top_k: 400
* nms/score_threshold: 0.005
* dataset_dir: dataset/vehicle

### 3. Accuracy

The accuracy of the model trained and evaluated on our private data is :

AP = 0.545 (IoU=.50:.05:.95)

AP = 0.764 (IoU=.50)

### 4. Inference

Users can employ the model for inference:

```
export CUDA_VISIBLE_DEVICES=0
python -u tools/infer.py -c configs/vehicle/vehicle_yolov3_darknet.yml \
                         -o weights=https://paddledet.bj.bcebos.com/models/vehicle_yolov3_darknet.pdparams \
                         --infer_dir configs/vehicle/demo \
                         --draw_threshold 0.2 \
                         --output_dir configs/vehicle/demo/output
```

Some inference results are visualized below:

![](../../docs/images/VehicleDetection_001.jpeg)

![](../../docs/images/VehicleDetection_005.png)
