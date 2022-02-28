English | [简体中文](README_cn.md)


# Releases

- 2021.11.03: <br>  
[2.3 version](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.3)<br>
Mobile object detection model ⚡[PP-PicoDet](configs/picodet)<br>  
Mobile keypoint detection model ⚡[PP-TinyPose](configs/keypoint/tiny_pose)<br>   
Real-time tracking system [PP-Tracking](deploy/pptracking)<br>  
Object detection models: [Swin-Transformer](configs/faster_rcnn), [TOOD](configs/tood), and [GFL](configs/gfl)<br>  
Tiny object detection optimization model [Sniper](configs/sniper)<br>  
Optimized model [PP-YOLO-EB](configs/ppyolo) for EdgeBoard<br>  
Mobile keypoint detection model [Lite HRNet](configs/keypoint), which supports the deployment of Paddle Lite

- 2021.08.10:
[2.2 version](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.2)<br> 
Object detection models with Transformers：[DETR](configs/detr), [Deformable DETR](configs/deformable_detr), and [Sparse RCNN](configs/sparse_rcnn)<br>  
[Keypoint detection](configs/keypoint) models using Dark, HRNet, and MPII dataset<br> 
[Head-tracking](configs/mot/headtracking21) and [vehicle-tracking](configs/mot/vehicle) multi-object tracking models.
- 2021.05.20: 
[2.1 version](https://github.com/PaddlePaddle/Paddleetection/tree/release/2.1)<br>
[Keypoint detection models](configs/keypoint): HigherHRNet and HRNet<br>
[Multi-Object tracking models](configs/mot): DeepSORT，JDE, and FairMOT<br>
Model compression for PPYOLO models<br>
Documents such as [EXPORT ONNX MODEL](deploy/EXPORT_ONNX_MODEL.md)


# Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle. It offers varied mainstream object detection, instance segmentation, tracking, and keypoint detection algorithms, provides access to configurable modules network components, data augmentation strategies, and loss functions. Also, it releases server-side and mobile SOTA industry practice models, integrates model compression and cross-platform high-performance deployment, in order to help developers complete the end-to-end development in a faster and better way.

### PaddleDetection makes it available to process images such as object detection, instance segmentation, multi-object tracking, and keypoint detection.

<div width="1000" align="center">
  <img src="docs/images/ppdet.gif"/>
</div>


### Features

- **Abundant Models**
PaddleDetection provides **100+ pre-trained models**, including **object detection**, **instance segmentation**, and **face detection** models. It covers a variety of **global competition champion** schemes.

- **Easy-to-use**
Components are in modular design. By decoupling network components, developers can easily build and test detection models and optimization strategies to obtain customized algorithms with high performance.

- **Production Ready**
Data augmentation, model construction, training, compression, and depolyment are integrated into an organic whole. Multi-architecture and multi-device deployment for **cloud and edge devices** has been further perfected.

- **High Performance**
Owing to the high-performance core of PaddlePaddle, the model training speed is fast and the memory occupation is small. FP16 training and multi-machine training are available as well.

#### Overview of Kit Structures

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
          <li><b>Object Detection</b></li>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>Libra RCNN</li>
            <li>Hybrid Task RCNN</li>
            <li>PSS-Det</li>
            <li>RetinaNet</li>
            <li>YOLOv3</li>
            <li>YOLOv4</li>  
            <li>PP-YOLOv1/v2</li>
            <li>PP-YOLO-Tiny</li>
            <li>SSD</li>
            <li>CornerNet-Squeeze</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
            <li>PP-PicoDet</li>
            <li>DETR</li>
            <li>Deformable DETR</li>
            <li>Swin Transformer</li>
            <li>Sparse RCNN</li>
        </ul>
        <li><b>Instance Segmentation</b></li>
        <ul>
            <li>Mask RCNN</li>
            <li>SOLOv2</li>
        </ul>
        <li><b>Face Detection</b></li>
        <ul>
            <li>FaceBoxes</li>
            <li>BlazeFace</li>
            <li>BlazeFace-NAS</li>
        </ul>
        <li><b>Multi-Object-Tracking</b></li>
        <ul>
            <li>JDE</li>
            <li>FairMOT</li>
            <li>DeepSort</li>
        </ul>
        <li><b>KeyPoint-Detection</b></li>
        <ul>
            <li>HRNet</li>
            <li>HigherHRNet</li>
        </ul>
      </ul>
      </td>
      <td>
        <ul>
          <li>ResNet(&vd)</li>
          <li>ResNeXt(&vd)</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Hourglass</li>
          <li>CBNet</li>
          <li>GCNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>VGG</li>
          <li>MobileNetv1/v3</li>  
          <li>GhostNet</li>
          <li>Efficientnet</li>  
          <li>BlazeNet</li>  
        </ul>
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>Non-local</li>
          </ul>  
        </ul>
        <ul><li><b>KeyPoint</b></li>
          <ul>
            <li>DarkPose</li>
          </ul>  
        </ul>
        <ul><li><b>FPN</b></li>
          <ul>
            <li>BiFPN</li>
            <li>BFP</li>  
            <li>HRFPN</li>
            <li>ACFPN</li>
          </ul>  
        </ul>  
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
          </ul>  
        </ul>  
        <ul><li><b>Post-processing</b></li>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul>  
        </ul>
        <ul><li><b>Speed</b></li>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul>  
        </ul>  
      </td>
      <td>
        <ul>
          <li>Resize</li>  
          <li>Lighting</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>Mosaic</li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
          <li>Random Perspective</li>  
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

#### Overview of Model Performance

The relationships between COCO mAP values and FPS values on Tesla V100 of representative models of server-side architectures and backbones are as follows.

<div align="center">
  <img src="docs/images/fps_map.png" />
  </div>

  **NOTE:**

  - `CBResNet stands` for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%.

  - `Cascade-Faster-RCNN` stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS on inference speed when COCO mAP is 47.8% of PaddleDetection models.

  - The COCO mAP of `PP-YOLO` is 45.9% and the FPS on Tesla V100 is 72.9. The two indicators both surpass those of [YOLOv4](https://arxiv.org/abs/2004.10934).

  - `PP-YOLO v2` is an optimized version of `PP-YOLO`, and its COCO mAP reaches 49.5% and its FPS on Tesla V100 achieves 68.9.

  - All these models are accessible in [Model Zoo](#ModelZoo)

The following diagram shows the relationship between COCO mAP values and FPS values of representative mobile models on Qualcomm Snapdragon 865.

<div align="center">
  <img src="docs/images/mobile_fps_map.png" width=600 />
</div>

**NOTE:**

- All data tested on Qualcomm Snapdragon 865(4\*A77 + 4\*A55) processor has a batch size of 1. The CPU has 4 threads, and the NCNN library are used in testing. The benchmark scripts is publiced at [MobileDetBenchmark](https://github.com/JiweiMaster/MobileDetBenchmark).
- [PP-PicoDet](configs/picodet) and [PP-YOLO-Tiny](configs/ppyolo) are developed and released by PaddleDetection, and other models are not available on PaddleDetection.

## Tutorials

### Quick Start Tutorials

- [Installation guide](docs/tutorials/INSTALL.md)
- [Dataset preparation](docs/tutorials/PrepareDataSet_en.md)
- [Quick start on PaddleDetection](docs/tutorials/GETTING_STARTED.md)
- [FAQ](docs/tutorials/FAQ)


### Advanced Tutorials

- Parameter configuration
  - [Parameter configuration for RCNN models](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation_en.md)
  - [Parameter configuration for PP-YOLO models](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation_en.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
  - [Prune/Quant/Distill](configs/slim)

- Inference and deployment
  - [Model export](deploy/EXPORT_MODEL_en.md)
  - [Paddle Inference](deploy/README_en.md)
      - [Python inference](deploy/python)
      - [C++ inference](deploy/cpp)
  - [Paddle-Lite](deploy/lite)
  - [Paddle Serving](deploy/serving)
  - [ONNX model export](deploy/EXPORT_ONNX_MODEL_en.md)
  - [Inference benchmark](deploy/BENCHMARK_INFER_en.md)

- Advanced development
  - [Data processing module](docs/advanced_tutorials/READER_en.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL_en.md)


## Model Zoo

- Universal object detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
  - [PP-YOLO](configs/ppyolo/README.md)
  - [PP-PicoDet](configs/picodet/README.md)
  - [Enhanced anchor free model--TTFNet](configs/ttfnet/README_en.md)
  - [Mobile models](static/configs/mobile/README_en.md)
  - [676 classes of object detection](static/docs/featured_model/LARGE_SCALE_DET_MODEL_en.md)
  - [Two-stage practical model--PSS-Det](configs/rcnn_enhance/README_en.md)
  - [SSLD pretrained models](docs/feature_models/SSLD_PRETRAINED_MODEL_en.md)
- Universal instance segmentation
  - [SOLOv2](configs/solov2/README.md)
- Rotated object detection
  - [S2ANet](configs/dota/README_en.md)
- [Keypoint detection](configs/keypoint)
  - [PP-TinyPose](configs/keypoint/tiny_pose)
  - HigherHRNet
  - HRNet
  - LiteHRNet
- [Multi-object tracking](configs/mot/README.md)
  - [PP-Tracking](deploy/pptracking/README.md)
  - [DeepSORT](configs/mot/deepsort/README.md)
  - [JDE](configs/mot/jde/README.md)
  - [FairMOT](configs/mot/fairmot/README.md)
- Vertical field
  - [Face detection](configs/face_detection/README_en.md)
  - [Pedestrian detection](configs/pedestrian/README.md)
  - [Vehicle detection](configs/vehicle/README.md)
- Competition schemes
  - [Objects365 2019 Challenge champion model](static/docs/featured_model/champion_model/CACascadeRCNN_en.md)
  - [Best single model of Open Images 2019-Object Detection](static/docs/featured_model/champion_model/OIDV5_BASELINE_MODEL_en.md)

## Applications

- [Automatic Christmas profile maker](static/application/christmas)
- [Android Fitness APP demo](https://github.com/zhiboniu/pose_demo_android)

## Updates

For updates, please refer to [change log](docs/CHANGELOG_en.md).


## License

The release of PaddleDetection is permitted by the [Apache 2.0 license](LICENSE).


## Contributions

Contributions are highly welcomed and we would appreciate your feedback!!
- Thanks [Mandroide](https://github.com/Mandroide) for cleaning the code and unifying some function interfaces.
- Thanks [FL77N](https://github.com/FL77N/) for providing the code of `Sparse-RCNN` model.
- Thanks [Chen-Song](https://github.com/Chen-Song) for offering the code of `Swin Faster-RCNN` model.
- Thanks [yangyudong](https://github.com/yangyudong2020) and [hchhtc123](https://github.com/hchhtc123) for developing the PP-Tracking GUI interface.
- Thanks [Shigure19](https://github.com/Shigure19) for developing the PP-TinyPose fitness APP.

## Citation

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
