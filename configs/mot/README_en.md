English | [简体中文](README.md)

# MOT (Multi-Object Tracking)

## Table of Contents
- [Introduction](#Introduction)
- [Installation](#Installation)
- [Model Zoo](#Model_Zoo)
- [Dataset Preparation](#Dataset_Preparation)
- [Citations](#Citations)

## Introduction
The mainstream multi-object tracking (MOT) algorithm of 'Tracking By Detecting' is mainly composed of two parts: detection and embedding. Detection detects the potential targets in each frame of the video. Embedding assigns and updates the detected objects to the corresponding tracks (named ReID). Due to their differences in implementation, MOT can be divided into **SDE** algorithms and **JDE** algorithms.

- **SDE** (Separate Detection and Embedding) algorithms separate detection from embedding. The most representative is **DeepSORT** algorithm. This solution can make the system fit any kind of detectors, and it can be optimized for each part separately. However, because the end-to-end process is slow and time-consuming, which poses a great challenge in the construction of a real-time MOT system.
- **JDE** (Joint Detection and Embedding) algorithms learn detection and embedding simultaneously in a shared neural network, and set the loss function with a multi-task learning approach. The representative ones are **JDE** and **FairMOT**. This solution juggle with precision and speed, achieving real-time MOT performance with high precision.

PaddleDetection makes three MOT algorithms of these two categories available: [DeepSORT](https://arxiv.org/abs/1812.00442) of SDE algorithms, and [JDE](https://arxiv.org/abs/1909.12605) and [FairMOT](https://arxiv.org/abs/2004.01888) of JDE algorithms.

### PP-Tracking Real-Time MOT System
In addition, PaddleDetection also provides [PP-Tracking](../../deploy/pptracking/README.md) real-time MOT system.
PP-Tracking is the first open-source real-time Multi-Object Tracking system, and it is based on the deep learning framework of PaddlePaddle. It has rich models, wide application scenarios, and efficient deployment.

PP-Tracking supports two paradigms: Mono-Camera Tracking (MOT) and Multi-Target Multi-Camera Tracking (MTMCT). To solve the difficulties and pain points of real business, PP-Tracking includes various application scenarios of Multi-Object Tracking such as pedestrian tracking, vehicle tracking, multi-class tracking, small object tracking, traffic statistics and multi-camera tracking. The deployment method supports the API call and GUI , and compatible deployment languages include Python and C++, The deployment platform environment supports Linux, NVIDIA Jetson, etc.

### Public Projects of AI studio
There is a public project of AI studio on PP-Tracking. You can refer to this [tutorial](https://aistudio.baidu.com/aistudio/projectdetail/3022582).

### Python Prediction and Deployment
PP-Tracking supports prediction and deployment with python. For the tutorial, refer to this [doc](../../deploy/pptracking/python/README.md).

### C++ Prediction and Deployment
PP-Tracking supports prediction and deployment with C++. For the tutorial, refer to this [doc](../../deploy/pptracking/cpp/README.md).

### GUI prediction and Deployment
PP-Tracking supports predict and deployment with GUI. Please refer to this [doc](https://github.com/yangyudong2020/PP-Tracking_GUi).

<div width="1000" align="center">
  <img src="../../docs/images/pptracking_en.png"/>
</div>

<div width="1000" align="center">
  <img src="../../docs/images/pptracking-demo.gif"/>
  <br>
  video source：VisDrone, BDD100K dataset</div>
</div>


## Dependency Installation
Install all the dependencies related to MOT:
```
pip install lap sklearn motmetrics openpyxl cython_bbox
or
pip install -r requirements.txt
```
**Note:**
- Install `cython_bbox` for Windows: `pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox`. You can refer to the [tutorial](https://stackoverflow.com/questions/60349980/is-there-a-way-to-install-cython-bbox-for-windows).
- Please make sure that [ffmpeg](https://ffmpeg.org/ffmpeg.html) is installed before prediction, on Linux(Ubuntu) platform you can achieve this with the following command:`apt-get update && apt-get install -y ffmpeg`.


## Model Zoo
- Basic Models
    - [DeepSORT](deepsort/README.md)
    - [JDE](jde/README.md)
    - [FairMOT](fairmot/README.md)
- Feature Models
    - [Pedestrian](pedestrian/README.md)
    - [Head](headtracking21/README.md)
    - [Vehicle](vehicle/README.md)
- Multi-Class Tracking
    - [MCFairMOT](mcfairmot/README.md)
- Multi-Target Multi-Camera Tracking
    - [MTMCT](mtmct/README.md)


## Dataset Preparation
### MOT Dataset
PaddleDetection have reproduced [JDE](https://github.com/Zhongdao/Towards-Realtime-MOT) and [FairMOT](https://github.com/ifzhang/FairMOT) with their shared dataset 'MIX', including **Caltech Pedestrian, CityPersons, CUHK-SYSU, PRW, ETHZ, MOT17 and MOT16**. The former six are used as federated datasets for training, and MOT16 is adopted as the test dataset. If you want to use them, please **follow their licenses**.

In order to train more models of different scenarios, their datasets should also be processed into the same format as the MIX dataset. Please refer to [doc](../../docs/tutorials/PrepareMOTDataSet.md) to prepare the datasets.

### Dataset Directory
First, download the image_lists.zip with the following command, and unzip it into `PaddleDetection/dataset/mot`:
```
wget https://dataset.bj.bcebos.com/mot/image_lists.zip
```

Then, download the MIX dataset using the following command, and unzip it into `PaddleDetection/dataset/mot`:
```
wget https://dataset.bj.bcebos.com/mot/MOT17.zip
wget https://dataset.bj.bcebos.com/mot/Caltech.zip
wget https://dataset.bj.bcebos.com/mot/CUHKSYSU.zip
wget https://dataset.bj.bcebos.com/mot/PRW.zip
wget https://dataset.bj.bcebos.com/mot/Cityscapes.zip
wget https://dataset.bj.bcebos.com/mot/ETHZ.zip
wget https://dataset.bj.bcebos.com/mot/MOT16.zip
```

The final directory is:
```
dataset/mot
  |——————image_lists
            |——————caltech.10k.val  
            |——————caltech.all  
            |——————caltech.train  
            |——————caltech.val  
            |——————citypersons.train  
            |——————citypersons.val  
            |——————cuhksysu.train  
            |——————cuhksysu.val  
            |——————eth.train  
            |——————mot16.train  
            |——————mot17.train  
            |——————prw.train  
            |——————prw.val
  |——————Caltech
  |——————Cityscapes
  |——————CUHKSYSU
  |——————ETHZ
  |——————MOT16
  |——————MOT17
  |——————PRW
```

### Data Format
All these datasets share the following structure:
```
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train
```
Annotations of these datasets are in a unified format. Every image has its annotated text. Given an image path, the annotation text path can be generated by replacing the string `images` with `labels_with_ids` and replacing `.jpg` with `.txt`.

In the annotated text, each line describes one bounding box:
```
[class] [identity] [x_center] [y_center] [width] [height]
```
**Note:**
- `class` is an id of classification, which starts from 0, and supports single class and multi-class.
- `identity` is an integer between `1` and `num_identities`(`num_identities` refers to the total number of instances of objects in the dataset). It would be `-1` without the annotation of `identity`.
- `[x_center] [y_center] [width] [height]` represent the coordinate of and center point, width and height. It should be noted that they are normalized by the width/height of the image, so they are floating point numbers ranging from 0 to 1.


## Citations
```
@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}

@article{wang2019towards,
  title={Towards Real-Time Multi-Object Tracking},
  author={Wang, Zhongdao and Zheng, Liang and Liu, Yixuan and Wang, Shengjin},
  journal={arXiv preprint arXiv:1909.12605},
  year={2019}
}

@article{zhang2020fair,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Chunyu and Wang, Xinggang and Zeng, Wenjun and Liu, Wenyu},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
```
