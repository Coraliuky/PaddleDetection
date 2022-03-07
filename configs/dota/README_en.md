[简体中文](README.md) | English
# S2ANet Model

## Content
- [S2ANet Model](#s2anet-model)
  - [Content](#content)
  - [Introduction](#introduction)
  - [Data Preparation](#data-preparation)
    - [DOTA Data](#dota-data)
    - [Data Customization](#data-customization)
  - [Starting Training](#starting-training)
    - [1. Installing IOU of the rotated box and calculating the OP](#1-installing-iou-of-the-rotated-box-and-calculating-the-op)
    - [2. Training](#2-training)
    - [3. Evaluation](#3-evaluation)
    - [4. Inference](#4-inference)
    - [5. DOTA Data Evaluation](#5-dota-data-evaluation)
  - [Model Library](#model-library)
    - [S2ANet Model](#s2anet-model-1)
  - [Inference and Deployment](#inference-and-deployment)
  - [Citations](#citations)

## Introduction

[S2ANet](https://arxiv.org/pdf/2008.09397.pdf) is used to detect rotated boxes, requiring to use PaddlePaddle 2.1.1(can be installed with PIP) or other [develop versions](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/install/Tables.html#whl-release).


## Data Preparation

### DOTA Data
[DOTA Dataset] is a dataset for object detection in aerial images, containing 2806 images with a resolution of 4000x4000.

|  Data version  |  Number of category  |   Number of images   |  Image size  |    Number of instances    |     Annotation method     |
|:--------:|:-------:|:---------:|:---------:| :---------:| :------------: |
|   v1.0   |   15    |   2806    | 800~4000  |   118282    |   OBB + HBB     |
|   v1.5   |   16    |   2806    | 800~4000  |   400000    |   OBB + HBB     |

Note: In OBB annotation, every object is annotated by an oriented bounding box, and the vertices are arranged in a clockwise order. In HBB style, every object is annotated by an horizontal bounding box.

There are 2,806 images in the DOTA dataset. 1,411 images make up a training set, and 458 images make up an evaluation set, and the remaining 937 images compose a test set.

If you need to segment the image data, please refer to the [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit).

After setting `crop_size=1024, stride=824, gap=200` parameters for data segmentation, there are 15,749 images in the training set, 5,297 images in the evaluation set, and 10,833 images in the test set.

### Data Customization

There are two annotation techniques:

- The first is rotated rectangular annotation with the use of the tool [roLabelImg](https://github.com/cgvict/roLabelImg).

- The second is quadrilateral annotation by converting them into outer rotated rectangles with the script, but this may lead to discrepancy between annotations and the reality.

Then convert the annotation result into the annotation format of COCO. Each `bbox` is in the format of `[x_center, y_center, width, height, angle]`, and the angle is expressed in radians.

You can refer to [spinal disk dataset](https://aistudio.baidu.com/aistudio/datasetdetail/85885), which is divided into the training set (230) and the test set (57). The address is: [spine_coco](https://paddledet.bj.bcebos.com/data/spine_coco.tar). The dataset has a small number of images for fast training of the S2ANet model.


## Starting Training

### 1. Installing IOU of the rotated box and calculating the OP

Using IOU of the rotated box to calculate [ext_op](../../ppdet/ext_op) refers to PaddlePaddle's [custom external operator](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/07_new_op/new_custom_op.html).

To use IOU of the rotated box to calculate the OP, the following conditions must be met:
- PaddlePaddle >= 2.1.1
- GCC == 8.2

Mirrored docker [paddle:2.1.1-gpu-cuda10.1-cudnn7](registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.1-cudnn7) is recommended here.

Run the following command to download the image and start the container:
```
sudo nvidia-docker run -it --name paddle_s2anet -v $PWD:/paddle --network=host registry.baidubce.com/paddlepaddle/paddle:2.1.1-gpu-cuda10.1-cudnn7 /bin/bash
```

If the paddle are installed in the mirror, start python3.7, and run the following code to check whether the paddle are installed properly:
```
import paddle
print(paddle.__version__)
paddle.utils.run_check()
```

enter `ppdet/ext_op`, and install：
```
python3.7 setup.py install
```

In Windows, perform the following steps for installation：

（1）Visual Studio (version required >= Visual Studio 2015 Update3) is required, and here will use VS2017 ;

（2）CLick to start --> Visual Studio 2017 --> X64 native tools command prompt for VS 2017;

（3）Set the environment variable：`set DISTUTILS_USE_SDK=1`

（4）Enter `PaddleDetection/ppdet/ext_op`，and use `python3.7 setup.py install` for installation.

After the installation, test whether compilng and calculation of the customized OP function normally:
```
cd PaddleDetecetion/ppdet/ext_op
python3.7 test.py
```

### 2. Training
**Note:**
In the configuration file, the learning rate is set based on the training of 8 GPUs. If the single-card GPU is used for training, set the learning rate to 1/8 of the original value.

Single-GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0
python3.7 tools/train.py -c configs/dota/s2anet_1x_spine.yml
```

Multiple-GPU Training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3.7 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/dota/s2anet_1x_spine.yml
```

You can use `--eval`to enable train-by-test.

### 3. Evaluation
```bash
python3.7 tools/eval.py -c configs/dota/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams

# Use a trained model for evaluation
python3.7 tools/eval.py -c configs/dota/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams
```
**Note:**
(1) In the DOTA dataset, train and val data are trained together as the training set, and the evaluation of the DOTA requires customized configuration of the dataset.

(2) Bone dataset is transformed from segmented data. As there is little difference between different types of disks in detection, and the score obtained by S2ANET algorithm is low, the default threshold  is 0.5 in evaluation, and a low mAP is normal. It is suggested to view the detection result through visualization.

### 4. Inference
Execute the following command, and image inference results will be saved in the `output` folder.
```bash
python3.7 tools/infer.py -c configs/dota/s2anet_1x_spine.yml -o weights=output/s2anet_1x_spine/model_final.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```
Perform inference by using provided models:
```bash
python3.7 tools/infer.py -c configs/dota/s2anet_1x_spine.yml -o weights=https://paddledet.bj.bcebos.com/models/s2anet_1x_spine.pdparams --infer_img=demo/39006.jpg --draw_threshold=0.3
```

### 5. DOTA Data Evaluation
Execute the following command, every piece of image inference results in `output` folder will be saved in a txt text with the same folder name.
```
python3.7 tools/infer.py -c configs/dota/s2anet_alignconv_2x_dota.yml -o weights=./weights/s2anet_alignconv_2x_dota.pdparams  --infer_dir=dota_test_images --draw_threshold=0.05 --save_txt=True --output_dir=output
```
Please refer to [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) to generate assessment files, whose format can refer to [DOTA Test](http://captain.whu.edu.cn/DOTAweb/tasks.html). Generate the zip file, and results of images of each class is stored in a txt file. Every row in the txt file follows the format: `image_id score x1 y1 x2 y2 x3 y3 x4 y4` and the files are uploaded to the server for evaluation. You can also refer to the `dataset/dota_coco/dota_generate_test_result.py` to generate assessment files and submit them to the server.

## Model Library

### S2ANet Model

|     Model     |  Conv Type  |   mAP    |   Model Download   |   Configuration File   |
|:-----------:|:----------:|:--------:| :----------:| :---------: |
|   S2ANet    |   Conv     |   71.42  |  [model](https://paddledet.bj.bcebos.com/models/s2anet_conv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dota/s2anet_conv_2x_dota.yml)                   |
|   S2ANet    |  AlignConv |   74.0   |  [model](https://paddledet.bj.bcebos.com/models/s2anet_alignconv_2x_dota.pdparams) | [config](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.3/configs/dota/s2anet_alignconv_2x_dota.yml)                   |

**Note:** `multiclass_nms` is used here, which is slightly different from NMS adopted by the original author.


## Inference and Deployment

The input of the `multiclass_nms` operator in Paddle supports quadrilateral inputs, so deployment can be done without the IOU operator of the rotated box.

Please refer to the deployment tutorial[Predict deployment](../../deploy/README_en.md).


## Citations
```
@article{han2021align,  
  author={J. {Han} and J. {Ding} and J. {Li} and G. -S. {Xia}},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  title={Align Deep Features for Oriented Object Detection},  
  year={2021},
  pages={1-11},  
  doi={10.1109/TGRS.2021.3062048}}

@inproceedings{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3974--3983},
  year={2018}
}
```
