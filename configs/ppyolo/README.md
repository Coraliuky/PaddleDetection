English | [简体中文](README_cn.md)

# PP-YOLO

## Contents
- [Introduction](#Introduction)
- [Model Zoo](#Model_Zoo)
- [Getting Start](#Getting_Started)
- [Future Work](#Future_Work)
- [Appendix](#Appendix)

## Introduction

[PP-YOLO](https://arxiv.org/abs/2007.12099) is an optimized version of YOLOv3 in PaddleDetection，and it outperforms [YOLOv4](https://arxiv.org/abs/2004.10934) on both COCO mAP and the inference spped。And it requires PaddlePaddle 2.0.2(available on pip now) or [Daily Version](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-develop).

The mAp(IoU=0.5:0.95) of PP-YOLO on COCO test-dev2017 dataset reaches 45.9% , its inference speed of FP32 on single V100 is 72.9 FPS, and that of FP16 with TensorRT on single V100 is 155.6 FPS.

<div align="center">
  <img src="../../docs/images/ppyolo_map_fps.png" width=500 />
</div>

The performance and the speed of YOLOv3 have been improved in PP-YOLO and PP-YOLOv2 improved by using the following methods:

- Better backbone: ResNet50vd-DCN
- Larger training batch size: 8 GPUs, and mini-batch size=24 on each GPU
- [Drop Block](https://arxiv.org/abs/1810.12890)
- [Exponential Moving Average](https://www.investopedia.com/terms/e/ema.asp)
- [IoU Loss](https://arxiv.org/pdf/1902.09630.pdf)
- [Grid Sensitive](https://arxiv.org/abs/2004.10934)
- [Matrix NMS](https://arxiv.org/pdf/2003.10152.pdf)
- [CoordConv](https://arxiv.org/abs/1807.03247)
- [Spatial Pyramid Pooling](https://arxiv.org/abs/1406.4729)
- Better ImageNet pretrained weights
- [PAN](https://arxiv.org/abs/1803.01534)
- Iou aware Loss
- larger input size

## Model Zoo

### PP-YOLO

|          Model           | Number of GPUs | Number of images/GPU |  Backbone  | Input size | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | Download | Config  |
|:------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: | :------: | :------: |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     608     |         44.8         |         45.2          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     512     |         43.9         |         44.4          |      89.9      |          188.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     416     |         42.1         |         42.5          |      109.1      |          215.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO                  |     8      |     24     | ResNet50vd |     320     |         38.9         |         39.3          |      132.2      |          242.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     608     |         45.3         |         45.9          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     512     |         44.4         |         45.0          |      89.9      |          188.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     416     |         42.7         |         43.2          |      109.1      |          215.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO_2x               |     8      |     24     | ResNet50vd |     320     |         39.5         |         40.1          |      132.2      |          242.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_2x_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_2x_coco.yml)                   |
| PP-YOLO               |     4      |     32     | ResNet18vd |     512     |         29.2         |         29.5          |      357.1      |          657.9          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLO               |     4      |     32     | ResNet18vd |     416     |         28.6         |         28.9          |      409.8      |          719.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLO               |     4      |     32     | ResNet18vd |     320     |         26.2         |         26.4          |      480.7      |          763.4          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r18vd_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r18vd_coco.yml)                   |
| PP-YOLOv2               |     8      |     12     | ResNet50vd |     640     |         49.1         |         49.5          |      68.9      |          106.5          | [model](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r50vd_dcn_365e_coco.yml)                   |
| PP-YOLOv2               |     8      |     12     | ResNet101vd |     640     |         49.7         |         50.3          |     49.5     |         87.0         | [model](https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolov2_r101vd_dcn_365e_coco.yml)                   |


**Note:**

- PP-YOLO is trained on COCO train2017 dataset and evaluated on val2017 & test-dev2017 datasets. The Box AP<sup>test</sup> refers to the evaluation result of `mAP(IoU=0.5:0.95)`.
- PP-YOLO uses 8 GPUs for training and mini-batch size on each GPU is 24. If the number of GPUs and the mini-batch size are changed, adjust the learning rate and iteration times according to [FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/FAQ.md).
- The inference speed of PP-YOLO inference is tesed on single Tesla V100 with the batch size of 1, CUDA 10.2, and CUDNN 7.5.1. The inference speed of TensorRT is tested with TensorRT 5.1.2.2.
- The FP32 inference speed test of PP-YOLO uses the inference model exported by `tools/export_model.py` and `--run_benchmark` of `depoly/python/infer.py`. And its benchmark test on the inference speed uses the Paddle inference library. All the test data do not contain those of preprocessing and output post-processing(NMS), which is the same as [YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet) in testing method.
- The TensorRT FP16 inference speed test excludes the time cost of bounding-box decoding(`yolo_box`) part compared with the FP32 test above, which means that data reading, bounding-box decoding and post-processing(NMS) are excluded(the same as [YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet))
- If you set `--run_benchmark=True`，you should install these dependencies at first, `pip install pynvml psutil GPUtil`.

### Mobile versions of PP-YOLO

|            Model             | Number of GPUs | Number of images/GPU | Model Size | Input shape | Box AP<sup>val</sup> |  Box AP50<sup>val</sup> | Kirin 990 1xCore(FPS) | Download | Config  |
|:----------------------------:|:-------:|:-------------:|:----------:| :-------:| :------------------: |  :--------------------: | :--------------------: | :------: | :------: |
| PP-YOLO_MobileNetV3_large    |    4    |      32       |    28MB    |   320    |         23.2         |           42.6          |           14.1         | [model](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_large_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_mbv3_large_coco.yml)                   |
| PP-YOLO_MobileNetV3_small    |    4    |      32       |    16MB    |   320    |         17.2         |           33.8          |           21.5         | [model](https://paddledet.bj.bcebos.com/models/ppyolo_mbv3_small_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_mbv3_small_coco.yml)                   |

**Note:**

- PP-YOLO_MobileNetV3 is trained on COCO train2017 datast and evaluated on val2017 dataset. The Box AP<sup>val</sup> refers to the evaluation result of `mAP(IoU=0.5:0.95)`,  and Box AP<sup>val</sup> is the evaluation result of `mAP(IoU=0.5)`.
- PP-YOLO_MobileNetV3 uses 4 GPUs for training with the mini-batch size of 32 on each GPU. If the number of GPUs and the mini-batch size are changed, learning rate and iteration times should be adjusted according to [FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.3/docs/tutorials/FAQ.md).
- The inference speed of PP-YOLO_MobileNetV3 is tested on Kirin 990 with 1 thread.

### PP-YOLO tiny

|            Model             | Number of GPUs | Number of images/GPU | Model Size | Post-quant Model Size | Input shape | Box AP<sup>val</sup> | Kirin 990 4xCore(FPS) | Download | Config | Quantized model |
|:----------------------------:|:-------:|:-------------:|:----------:| :-------------------: | :---------: | :------------------: | :-------------------: | :------: | :----: | :--------------: |
| PP-YOLO tiny                 |    8    |      32       |   4.2MB    |       **1.3M**        |     320     |         20.6         |          92.3         | [model](https://paddledet.bj.bcebos.com/models/ppyolo_tiny_650e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_tiny_650e_coco.yml)  | [inference model](https://paddledet.bj.bcebos.com/models/ppyolo_tiny_quant.tar) |
| PP-YOLO tiny                 |    8    |      32       |   4.2MB    |       **1.3M**        |     416     |         22.7         |          65.4         | [model](https://paddledet.bj.bcebos.com/models/ppyolo_tiny_650e_coco.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_tiny_650e_coco.yml)  | [inference model](https://paddledet.bj.bcebos.com/models/ppyolo_tiny_quant.tar) |

**Note:**

- PP-YOLO-tiny is trained on COCO train2017 datast and evaluated on val2017 dataset，Box AP<sup>val</sup> is the evaluation result of `mAP(IoU=0.5:0.95)`, and Box AP<sup>val</sup> is the evaluation result of `mAP(IoU=0.5)`.
- PP-YOLO-tiny uses 8 GPUs for training with the mini-batch size of 32 on each GPU. If the number of GPUs and the mini-batch size are changed, the learning rate and iteration times should be adjusted according to [FAQ](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/docs/tutorials/FAQ/README.md).
- PP-YOLO-tiny inference speed is tested on Kirin 990 with 4 threads by arm8.
- It is also available to access post-quant model compression of PP-YOLO-tiny, which can compress the model size to **1.3MB** with little impact on the inference speed and the precision.

### PP-YOLO on Pascal VOC

PP-YOLO models trained on Pascal VOC dataset:

|       Model        | Number of GPUs | Number of images/GPU |  Backbone  | Input shape | Box AP50<sup>val</sup> | Download | Config  |
|:------------------:|:----------:|:----------:|:----------:| :----------:| :--------------------: | :------: | :-----: |
| PP-YOLO            |    8    |       12      | ResNet50vd |     608     |          84.9          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_voc.yml)                   |
| PP-YOLO            |    8    |       12      | ResNet50vd |     416     |          84.3          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_voc.yml)                   |
| PP-YOLO            |    8    |       12      | ResNet50vd |     320     |          82.2          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/ppyolo/ppyolo_r50vd_dcn_voc.yml)                   |

## Getting Started

### 1. Training

Train PP-YOLO on 8 GPUs with the following commands (all the commands should be run under the root directory of PaddleDetection as default). And start the alternate assessment by setting `--eval`.

```bash
python -m paddle.distributed.launch --log_dir=./ppyolo_dygraph/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml &>ppyolo_dygraph.log 2>&1 &
```

Optional: Run `tools/anchor_cluster.py` to get anchors for your dataset, and modify the anchor setting in model configuration files and reader configuration files, such as those in `configs/ppyolo/_base_/ppyolo_tiny.yml` and `configs/ppyolo/_base_/ppyolo_tiny_reader.yml`.

``` bash
python tools/anchor_cluster.py -c configs/ppyolo/ppyolo_tiny_650e_coco.yml -n 9 -s 320 -m v2 -i 1000
```

### 2. Evaluation

Run the following demands to evaluate PP-YOLO on COCO val2017 dataset with single-GPU.

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams

# use checkpoints saved in the training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=output/ppyolo_r50vd_dcn_1x_coco/model_final
```

`configs/ppyolo/ppyolo_test.yml` is available to assess COCO test-dev2017. Please download COCO test-dev2017 dataset from [COCO datasets download](https://cocodataset.org/#download), decompress it to path configured by `EvalReader.dataset` of `configs/ppyolo/ppyolo_test.yml`, and run the evaluation by following commands:

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_test.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams

# use checkpoints saved in the training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_test.yml -o weights=output/ppyolo_r50vd_dcn_1x_coco/model_final
```

The result will be saved in `bbox.json`, and it will be compressed into a `zip` package and uploaded to [COCO dataset evaluation](https://competitions.codalab.org/competitions/20794#participate) to evaluate.

**NOTE 1:** `configs/ppyolo/ppyolo_test.yml` is only used for evaluating COCO test-dev2017 dataset, but not for the training and evaluation of COCO val2017 dataset.

**NOTE 2:** Due to the overall upgrade of the dynamic graph framework, the following weight models published by PaddleDetection need to add the -- bias field in the evaluation. For example,

```bash
# use weights released in PaddleDetection model zoo
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --bias
```
These models are:

1.ppyolo_r50vd_dcn_1x_coco

2.ppyolo_r50vd_dcn_voc

3.ppyolo_r18vd_coco

4.ppyolo_mbv3_large_coco

5.ppyolo_mbv3_small_coco

6.ppyolo_tiny_650e_coco

### 3. Inference

Perform image inference in single GPU with following commands, including using `--infer_img` to designate the path or using `--infer_dir` to infer all images in a designated directory.

```bash
# Inference of single image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439_640x640.jpg

# inference of all images in the directory
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_dir=demo
```

### 4. Inferece and Deployment

For the benchmark inference and deployment, export the model exported with `tools/export_model.py` and perform it with the Paddle inference library. For example, you can run the following commands:

```bash
# export the model, which will be save in output/ppyolo as default
python tools/export_model.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams

# inference with the Paddle Inference library
CUDA_VISIBLE_DEVICES=0 python deploy/python/infer.py --model_dir=output_inference/ppyolo_r50vd_dcn_1x_coco --image_file=demo/000000014439_640x640.jpg --device=GPU
```


## Appendix

Ablation experiment results of the optimization of PP-YOLO based on YOLOv3 are as follows.

| NO.  |        Model                 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | Params(M) | FLOPs(G) | V100 FP32 FPS |
| :--: | :--------------------------- | :------------------: |:--------------------: | :-------: | :------: | :-----------: |
|  A   | YOLOv3-DarkNet53             |         38.9         |           -           |   59.13   |  65.52   |      58.2     |
|  B   | YOLOv3-ResNet50vd-DCN        |         39.1         |           -           |   43.89   |  44.71   |      79.2     |
|  C   | B + LB + EMA + DropBlock     |         41.4         |           -           |   43.89   |  44.71   |      79.2     |
|  D   | C + IoU Loss                 |         41.9         |           -           |   43.89   |  44.71   |      79.2     |
|  E   | D + IoU Aware                |         42.5         |           -           |   43.90   |  44.71   |      74.9     |
|  F   | E + Grid Sensitive           |         42.8         |           -           |   43.90   |  44.71   |      74.8     |
|  G   | F + Matrix NMS               |         43.5         |           -           |   43.90   |  44.71   |      74.8     |
|  H   | G + CoordConv                |         44.0         |           -           |   43.93   |  44.76   |      74.1     |
|  I   | H + SPP                      |         44.3         |         45.2          |   44.93   |  45.12   |      72.9     |
|  J   | I + Better ImageNet Pretrain |         44.8         |         45.2          |   44.93   |  45.12   |      72.9     |
|  K   | J + 2x Scheduler             |         45.3         |         45.9          |   44.93   |  45.12   |      72.9     |

**Note:**

- Performance and inference speed are measured with the input shape of 608.
- All models are trained on COCO train2017 dataset and evaluated on val2017 and test-dev2017 datasets，`Box AP` is the evaluation result of `mAP(IoU=0.5:0.95)`.
- Inference speed is tested on single Tesla V100 with the batch size of 1 by using the above benchmark test method, and the environment configuration is CUDA 10.2 and CUDNN 7.5.1.
- [YOLOv3-DarkNet53](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/yolov3/yolov3_darknet53_270e_coco.yml) with the mAP of 39.0 is an optimized version of YOLOv3 model in PaddleDetection. Visit [YOLOv3](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/configs/yolov3/README.md) for details.

## Citation

```
@article{huang2021pp,
  title={PP-YOLOv2: A Practical Object Detector},
  author={Huang, Xin and Wang, Xinxin and Lv, Wenyu and Bai, Xiaying and Long, Xiang and Deng, Kaipeng and Dang, Qingqing and Han, Shumin and Liu, Qiwen and Hu, Xiaoguang and others},
  journal={arXiv preprint arXiv:2104.10419},
  year={2021}
}
@misc{long2020ppyolo,
title={PP-YOLO: An Effective and Efficient Implementation of Object Detector},
author={Xiang Long and Kaipeng Deng and Guanzhong Wang and Yang Zhang and Qingqing Dang and Yuan Gao and Hui Shen and Jianguo Ren and Shumin Han and Errui Ding and Shilei Wen},
year={2020},
eprint={2007.12099},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
