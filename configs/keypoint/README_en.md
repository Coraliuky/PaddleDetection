[简体中文](README.md) | English

# KeyPoint Detection Models



## Introduction

- The keypoint detection part of PaddleDetection keeps pace with the state-of-the-art algorithm. It includes Top-Down and Bottom-Up solutions to meet various needs of users.

<div align="center">
  <img src="./football_keypoint.gif" width='800'/>
</div>



####   Model Zoo
COCO Dataset
| Model              | Input Size | AP(coco val) |                           Download                           | Config                                                    |
| :---------------- | -------- | :----------: | :----------------------------------------------------------: | ----------------------------------------------------------- |
| HigherHRNet-w32       | 512      |     67.1     | [higherhrnet_hrnet_w32_512.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512.yml)       |
| HigherHRNet-w32       | 640      |     68.3     | [higherhrnet_hrnet_w32_640.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_640.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_640.yml)       |
| HigherHRNet-w32+SWAHR | 512      |     68.9     | [higherhrnet_hrnet_w32_512_swahr.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/higherhrnet_hrnet_w32_512_swahr.pdparams) | [config](./higherhrnet/higherhrnet_hrnet_w32_512_swahr.yml) |
| HRNet-w32             | 256x192  |     76.9     | [hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x192.pdparams) | [config](./hrnet/hrnet_w32_256x192.yml)                     |
| HRNet-w32             | 384x288  |     77.8     | [hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_384x288.pdparams) | [config](./hrnet/hrnet_w32_384x288.yml)                     |
| HRNet-w32+DarkPose             | 256x192  |     78.0     | [dark_hrnet_w32_256x192.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_256x192.pdparams) | [config](./hrnet/dark_hrnet_w32_256x192.yml)                     |
| HRNet-w32+DarkPose             | 384x288  |     78.3     | [dark_hrnet_w32_384x288.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/dark_hrnet_w32_384x288.pdparams) | [config](./hrnet/dark_hrnet_w32_384x288.yml)                     |
| WiderNaiveHRNet-18         | 256x192  |     67.6(+DARK 68.4)     | [wider_naive_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/wider_naive_hrnet_18_256x192_coco.pdparams) | [config](./lite_hrnet/wider_naive_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   | 256x192  |     66.5     | [lite_hrnet_18_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_256x192_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_18_256x192_coco.yml)     |
| LiteHRNet-18                   | 384x288  |     69.7     | [lite_hrnet_18_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_18_384x288_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_18_384x288_coco.yml)     |
| LiteHRNet-30                   | 256x192  |     69.4     | [lite_hrnet_30_256x192_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_256x192_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_30_256x192_coco.yml)     |
| LiteHRNet-30                   | 384x288  |     72.5     | [lite_hrnet_30_384x288_coco.pdparams](https://bj.bcebos.com/v1/paddledet/models/keypoint/lite_hrnet_30_384x288_coco.pdparams) | [config](./lite_hrnet/lite_hrnet_30_384x288_coco.yml)     |


Note：The AP results of Top-Down models are based on Ground Truth bounding boxes.

MPII Dataset
| Model  | Input Size | PCKh(Mean) | PCKh(Mean@0.1) |                           Download                           | Config                                     |
| :---- | -------- | :--------: | :------------: | :----------------------------------------------------------: | -------------------------------------------- |
| HRNet-w32 | 256x256  |    90.6    |      38.5      | [hrnet_w32_256x256_mpii.pdparams](https://paddledet.bj.bcebos.com/models/keypoint/hrnet_w32_256x256_mpii.pdparams) | [config](./hrnet/hrnet_w32_256x256_mpii.yml) |


We have also released [PP-TinyPose](./tiny_pose/README_en.md), a real-time keypoint detection model optimized for mobile devices. Just explore it.

## Getting Started

### 1. Environment Installation

​    Please refer to [PaddleDetection Installation Guide](../../docs/tutorials/INSTALL.md) to install PaddlePaddle and PaddleDetection.

### 2. Dataset Preparation

​    Currently, KeyPoint Detection Models support [COCO](https://cocodataset.org/#keypoints-2017) and [MPII](http://human-pose.mpi-inf.mpg.de/#overview) datasets. For data preparation, refer to [Keypoint Dataset Preparation](../../docs/tutorials/PrepareKeypointDataSet_en.md).

​   For description of config files, please refer to [Keypoint Config Guide](../../docs/tutorials/KeyPointConfigGuide_en.md).

  - Note that, when testing by detected bounding boxes in Top-Down method, We should get `bbox.json` by a detection model. You can download the detected results for COCO val2017 [(Detector having human AP of 56.4 on COCO val2017 dataset)](https://paddledet.bj.bcebos.com/data/bbox.json) directly, put it at the root path (`PaddleDetection/`), and set `use_gt_bbox: False` in config file.

### 3、Training and Testing

​    **Single-GPU Training:**

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml
```

​    **Multiple-GPU Training:**

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m paddle.distributed.launch tools/train.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml
```

​    **Evaluation**

```shell
#COCO DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml

#MPII DataSet
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/hrnet/hrnet_w32_256x256_mpii.yml

#If you only need to save the prediction result, you can set --save_prediction_only. Then the result will be saved in output/keypoints_results.json by default.
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml --save_prediction_only
```

​    **Inference**

​    Note：Top-down models can only infer cropped images containing one person. If you want to infer images containing several people, please use "joint deployment and inference". Or you can adopt a Bottom-up model.

```shell
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=./output/higherhrnet_hrnet_w32_512/model_final.pdparams --infer_dir=../images/ --draw_threshold=0.5 --save_txt=True
```

​    **Deployment and Inference**

```shell
#Model export
python tools/export_model.py -c configs/keypoint/higherhrnet/higherhrnet_hrnet_w32_512.yml -o weights=output/higherhrnet_hrnet_w32_512/model_final.pdparams

#Deployment and inference
#Deploy inference on keypoint-based top down/bottom up models respectively. In this mode, top-down models only support inference for cropped images containing one person.
python deploy/python/keypoint_infer.py --model_dir=output_inference/higherhrnet_hrnet_w32_512/ --image_file=./demo/000000014439_640x640.jpg --device=gpu --threshold=0.5
python deploy/python/keypoint_infer.py --model_dir=output_inference/hrnet_w32_384x288/ --image_file=./demo/hrnet_demo.jpg --device=gpu --threshold=0.5

#Joint deployment and inference of detector and keypoint-based top-down models.
python deploy/python/det_keypoint_unite_infer.py --det_model_dir=output_inference/ppyolo_r50vd_dcn_2x_coco/ --keypoint_model_dir=output_inference/hrnet_w32_384x288/ --video_file=../video/xxx.mp4  --device=gpu
```

​    **Joint deployment and inference with Multi-Object Tracking model FairMOT**

```shell
#FairMOT model export
python tools/export_model.py -c configs/mot/fairmot/fairmot_dla34_30e_1088x608.yml -o weights=https://paddledet.bj.bcebos.com/models/mot/fairmot_dla34_30e_1088x608.pdparams

#Joint inference with the exported FairMOT model and the keypoint-based model
python deploy/python/mot_keypoint_unite_infer.py --mot_model_dir=output_inference/fairmot_dla34_30e_1088x608/ --keypoint_model_dir=output_inference/higherhrnet_hrnet_w32_512/ --video_file={your video name}.mp4 --device=GPU
```
**Note:**
 To export MOT model, please refer to [this](../../configs/mot/README_en.md).


## Citations
```
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}

@InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Feng and Zhu, Xiatian and Dai, Hanbin and Ye, Mao and Zhu, Ce},
    title = {Distribution-Aware Coordinate Representation for Human Pose Estimation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}
```
