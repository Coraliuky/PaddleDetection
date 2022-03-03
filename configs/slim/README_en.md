# Model Compression

In PaddleDetection, a complete tutorial and benchmarks for model compression based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) are accessible. Currently supported methods:

- [Pruning](prune)
- [Quantization](quant)
- [Distillation](distill)
- [The joint strategy](extensions)

It is recommended that you use a combination of prunning and distillation, or use pruning, QTA, and PTQ for detection model compression. The following part will take YOLOv3 as an example to carry out pruning, distillation and quantization experiments.

## Experimental Environment

- Python 3.7+
- PaddlePaddle >= 2.1.0
- PaddleSlim >= 2.1.0
- CUDA 10.1+
- cuDNN >=7.6.5

**Connection between PaddleDetection, Paddle and PaddleSlim Version**
| PaddleDetection Version | PaddlePaddle Version | PaddleSlim Version |                                                                                               Note                                                                                               |
| :---------------------: | :------------------: | :----------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       release/2.3       |       >= 2.1         |        2.1         | Post-training quantization depends on Paddle 2.2 and PaddleSlim 2.2  |
|       release/2.1       |       2.2            |        >= 2.1.0    | 2.1 |
|       release/2.0       |       >= 2.0.1       |        2.0         |   Quantization depends on Paddle 2.1 and PaddleSlim 2.1           |


#### PaddleSlim Installation
- Method 1: Direct Installation：
```
pip install paddleslim -i https://pypi.tuna.tsinghua.edu.cn/simple
```
- Method 2: Compiling and installation：
```
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
python setup.py install
```

## Quick Start

### Training

```shell
python tools/train.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml}
```

- `-c`: the configuration file of a specified model
- `--slim_config`: the configuration file of a specified compression technique


### Evaluation

```shell
python tools/eval.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} -o weights=output/{SLIM_CONFIG}/model_final
```

- `-c`: the configuration file of a specified model
- `--slim_config`: the configuration file of a specified compression technique
- `-o weights`: the path of a model trained by a specified compression algorithm.

### Testing

```shell
python tools/infer.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} \
    -o weights=output/{SLIM_CONFIG}/model_final
    --infer_img={IMAGE_PATH}
```

- `-c`: the configuration file of a specified model
- `--slim_config`: the configuration file of a specified compression technique
- `-o weights`: the path of a model trained by a specified compression algorithm
- `--infer_img`: the path of a specified test image


## Full-Chain Deployment

### Dynamic-to-Static Model Export

```shell
python tools/export_model.py -c configs/{MODEL.yml} --slim_config configs/slim/{SLIM_CONFIG.yml} -o weights=output/{SLIM_CONFIG}/model_final
```

- `-c`: the configuration file of a specified model
- `--slim_config`: the configuration file of a specified compression technique
- `-o weights`: the path of a model trained by a specified compression algorithm

### Deployment and Inference

- Paddle-Inference：
    - [Python Deployment](../../deploy/python/README.md)
    - [C++ Deployment](../../deploy/cpp/README.md)
    - [TensorRT Inference and Deployment Tutorial](../../deploy/TENSOR_RT.md)
- Server-side deployment: uses [PaddleServing](../../deploy/serving/README.md)
- Mobile deployment: uses [Paddle-Lite](../../deploy/lite/README.md)

## Benchmark

### Pruning

#### Benchmark of Pascal VOC

|       Model        | Compression technique  |     GFLOPs    | Model Volume(MB) | Input Size | Delay Prediction (SD855) |   Box AP   |                                              Download                                              |                                                      Model Configuration File                                                      |                                              Compression Algorithm Configuration File                                               |
| :----------------: | :-------------------: | :------------: | :--------------: | :--------: | :-----------------------: | :--------: | :------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv3-MobileNetV1 |       baseline        |     24.13      |        93        |    608     |       332.0ms        |    75.1    |        [Link](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams)        | [Configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) |                                                                  -                                                                  |
| YOLOv3-MobileNetV1 | Pruning-l1_norm(sensity) | 15.78(-34.49%) |     66(-29%)     |    608     |          -           | 78.4(+3.3) | [Link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_voc_prune_l1_norm.pdparams) | [Configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml) | [Configuration file of slim](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/prune/yolov3_prune_l1_norm.yml) |

#### Benchmark of COCO
|           Mode            | Compression Technique | GFLOPs | Model Volume(MB) | Input Size | Delay Prediction (SD855) | Box AP |                 Download                                                                                        |    Model Configuration File    |      Compression Algorithm Configuration File                                                |
| :-----------------------: | :------------------: | :----: | :--------------: | :--------: | :----------------------: | :----: | :-------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
| PP-YOLO-MobileNetV3_large |       baseline       |   --   |       18.5       |    608        |            25.1ms        |  23.2  |      [link](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams)       |   [configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_mbv3_large_coco.yml)   |                                                                      -                                                                      |
| PP-YOLO-MobileNetV3_large |      Pruning-FPGM       |  -37%  |       12.6       |    608     |          -           |  22.3  | [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_prune_fpgm.pdparams) |   [configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_mbv3_large_coco.yml)   | [slim configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/prune/ppyolo_mbv3_large_prune_fpgm.yml) |
|     YOLOv3-DarkNet53      |       baseline       |   --   |      238.2       |    608     |          -           |  39.0  |      [link](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams)       | [configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml) |                                                                      -                                                                      |
|     YOLOv3-DarkNet53      |      Pruning-FPGM       |  -24%  |        -         |    608     |          -           |  37.6  |  [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_prune_fpgm.pdparams)   | [configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml) |  [slim configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/prune/yolov3_darknet_prune_fpgm.yml)   |
|       PP-YOLO_R50vd       |       baseline       |   --   |      183.3       |    608     |          -           |  44.8  |     [link](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams)      |  [configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)  |                                                                      -                                                                      |
|       PP-YOLO_R50vd       |      Pruning-FPGM       |  -35%  |        -         |    608     |          -           |  42.1  |   [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_prune_fpgm.pdparams)    |  [configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)  |   [slim configuration file](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/prune/ppyolo_r50vd_prune_fpgm.yml)    |

Description:
- Currently, pruning can be applied to all models except the RCNN series.
- The dealy prediction on SD855 is deployed by using PaddleLite, and performed by using the ARM8 architecture and 4 threads.

### Quantization

#### COCO Benchmark

| Model                     | Compression technique       | Input size  | Model volume(MB) | Delay prediction (V100) | Delay prediction (SD855) |        Box AP         |                                          Download                                           |                                 Download of the inference model                                 |                                                          Model's configuration file                                                          |                                                Configuration file of the compression algorithm                                            |
| ------------------------- | -------------------------- | ----------- | :--------------: | :--------------------: | :---------------------: | :-------------------: | :-----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------: |
| PP-YOLOv2_R50vd           | baseline                   | 640         |      208.6       |         19.1ms         |           --            |         49.1          |    [link](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams)     |    [link](https://paddledet.bj.bcebos.com/models/slim/ppyolov2_r50vd_dcn_365e_coco.tar)     |      [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)       |                                                                    -                                                                     |
| PP-YOLOv2_R50vd           | PACT quantization   | 640         |        --        |         17.3ms         |           --            |         48.1          |     [link](https://paddledet.bj.bcebos.com/models/slim/ppyolov2_r50vd_dcn_qat.pdparams)     |       [link](https://paddledet.bj.bcebos.com/models/slim/ppyolov2_r50vd_dcn_qat.tar)        |      [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)       |    [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/ppyolov2_r50vd_dcn_qat.yml)     |
| PP-YOLO_R50vd             | baseline                   | 608         |      183.3       |         17.4ms         |           --            |         44.8          |      [link](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams)       |      [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_dcn_1x_coco.tar)       |      [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)       |                                                                    -                                                                     |
| PP-YOLO_R50vd             | PACT quantization    | 608         |       67.3       |         13.8ms         |           --            |         44.3          |     [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_qat_pact.pdparams)      |        [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_r50vd_qat_pact.tar)        |      [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)       |     [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/ppyolo_r50vd_qat_pact.yml)     |
| PP-YOLO-MobileNetV3_large | baseline                   | 320         |       18.5       |         2.7ms          |         27.9ms          |         23.2          |       [link](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams)        |       [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_coco.tar)        |       [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_mbv3_large_coco.yml)        |                                                                    -                                                                     |
| PP-YOLO-MobileNetV3_large | Common QAT | 320         |       5.6        |           --           |         25.1ms          |         24.3          |     [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_qat.pdparams)      |        [link](https://paddledet.bj.bcebos.com/models/slim/ppyolo_mbv3_large_qat.tar)        |       [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_mbv3_large_coco.yml)        |     [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/ppyolo_mbv3_large_qat.yml)     |
| YOLOv3-MobileNetV1        | baseline                   | 608         |       94.2       |         8.9ms          |          332ms          |         29.4          |    [link](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams)    |    [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_270e_coco.tar)    |    [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml)    |                                                                    -                                                                     |
| YOLOv3-MobileNetV1        | Common QAT | 608         |       25.4       |         6.6ms          |          248ms          |         30.5          |  [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_qat.pdparams)  |    [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_qat.tar)     |    [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml)    | [slim Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/yolov3_mobilenet_v1_qat.yml)  |
| YOLOv3-MobileNetV3        | baseline                   | 608         |       90.3       |         9.4ms          |         367.2ms         |         31.4          | [link](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) | [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_large_270e_coco.tar) | [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) |                                                                    -                                                                     |
| YOLOv3-MobileNetV3        | PACT Online quantitative   | 608         |       24.4       |         8.0ms          |         280.0ms         |         31.1          |  [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_coco_qat.pdparams)  |    [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v3_coco_qat.tar)     | [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v3_large_270e_coco.yml) | [slim Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/yolov3_mobilenet_v3_qat.yml)  |
| YOLOv3-DarkNet53          | baseline                   | 608         |      238.2       |         16.0ms         |           --            |         39.0          |     [link](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams)      |     [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet53_270e_coco.tar)      |     [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml)      |                                                                    -                                                                     |
| YOLOv3-DarkNet53          | Common QAT | 608         |       78.8       |         12.4ms         |           --            |         38.8          |    [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_coco_qat.pdparams)     |       [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_darknet_coco_qat.tar)       |     [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml)      |    [slim Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/yolov3_darknet_qat.yml)    |
| SSD-MobileNet_v1          | baseline                   | 300         |       22.5       |         4.4ms          |         26.6ms          |         73.8          |    [link](https://paddledet.bj.bcebos.com/models/ssd_mobilenet_v1_300_120e_voc.pdparams)    |    [link](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_120e_voc.tar)    |     [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml)      |                                                                    -                                                                     |
| SSD-MobileNet_v1          | Common QAT | 300         |       7.1        |           --           |         21.5ms          |         72.9          |  [link](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_voc_qat.pdparams)  |    [link](https://paddledet.bj.bcebos.com/models/slim/ssd_mobilenet_v1_300_voc_qat.tar)     |     [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml)      |   [slim Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/ssd_mobilenet_v1_qat.yml)   |
| Mask-ResNet50-FPN         | baseline                   | (800, 1333) |      174.1       |        359.5ms         |           --            |       39.2/35.6       |      [link](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_1x_coco.pdparams)      |      [link](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_coco.tar)      |    [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml)     |                                                                    -                                                                     |
| Mask-ResNet50-FPN         | Common QAT | (800, 1333) |        --        |           --           |           --            | 39.7(+0.5)/35.9(+0.3) |    [link](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_qat.pdparams)    |      [link](https://paddledet.bj.bcebos.com/models/slim/mask_rcnn_r50_fpn_1x_qat.tar)       |    [Configuration File ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml)     | [slim configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/quant/mask_rcnn_r50_fpn_1x_qat.yml) |

Description:
- Non-quantified models of the delay prediction on V100 are tested by TensorRT FP32, and the quantified models are tested by TensorRT INT8, and both of them include NMS.
- The delay prediction on SD855 are deployed by PaddleLite, and performed by using the ARM8 architecture and 4 threads.

### Post-Training Quantization
Prepare the val dataset to calibrate the PTQ models:

```shell
python tools/post_quant.py -c configs/{MODEL.yml} --slim_config configs/slim/post_quant/{SLIM_CONFIG.yml}
```
For example:
```shell
python3.7 tools/post_quant.py -c configs/ppyolo/ppyolo_mbv3_large_coco.yml --slim_config=configs/slim/post_quant/ppyolo_mbv3_large_ptq.yml
```

### Distillation

#### Benchmark of COCO

| Model              | Compression technique | Input size |   Box AP   |                                           Download                                            |                                                       Model's configuration file                                                       |                                                     Configuration file of the compression technique                                                      |
| ------------------ | -------------------- | ---------- | :--------: | :-------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline             | 608        |    29.4    |     [link](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams)     | [Configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                                                                         -                                                                         |
| YOLOv3-MobileNetV1 | Distillation         | 608        | 31.0(+1.6) | [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_distill.pdparams) | [Configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/distill/yolov3_mobilenet_v1_coco_distill.yml) |

- Please refer to the document [Distillation methods](distill/README.md) for distillation methods.

### Combination of Distillation and Pruning

#### Benchmark of COCO

| Model              | Compression technique     | Input size |    GFLOPs    | Model Volume(MB) | Delay Prediction (SD855) |   Box AP   |                                              Download                                               |                                                       Model's configuration file                                                       |                                                      Configuration file of the compression algorithm                                                         |
| ------------------ | ------------------------ | ---------- | :----------: | :--------------: | :---------------------: | :--------: | :-------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------: |
| YOLOv3-MobileNetV1 | baseline                 | 608        |    24.65     |       94.2       |         332.0ms         |    29.4    |        [link](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams)        | [Configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) |                                                                             -                                                                              |
| YOLOv3-MobileNetV1 | Distillation + Pruning | 608        | 7.54(-69.4%) |   30.9(-67.2%)   |         166.1ms         | 28.4(-1.0) | [link](https://paddledet.bj.bcebos.com/models/slim/yolov3_mobilenet_v1_coco_distill_prune.pdparams) | [Configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml) | [slim configuration file ](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/slim/extensions/yolov3_mobilenet_v1_coco_distill_prune.yml) |
