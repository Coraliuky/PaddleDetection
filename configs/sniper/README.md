English | [简体中文](README_cn.md)

# SNIPER: Efficient Multi-Scale Training

## Model Zoo

| With/without sniper   | Number of GPUs    | Number of images/GPU |    Model  |    Dataset     | Learning rate strategy | Box AP |          Download                  | Config |
| :---------------- | :-------------------: | :------------------: | :-----: | :-----: | :------------: | :-----: | :-----------------------------------------------------: | :-----: |
| w/o   |    4    |    1    | ResNet-r50-FPN      | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)  |   1x    |  23.3  | [Download link](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_fpn_1x_visdrone.pdparams ) | [Config](./faster_rcnn_r50_fpn_1x_visdrone.yml) |
| w/ |    4    |    1    | ResNet-r50-FPN      | [VisDrone](https://github.com/VisDrone/VisDrone-Dataset)   |   1x    |  29.7  | [Download link](https://bj.bcebos.com/v1/paddledet/models/faster_rcnn_r50_fpn_1x_sniper_visdrone.pdparams) | [Config](./faster_rcnn_r50_fpn_1x_sniper_visdrone.yml) |


### Note
> Here, we use THE VisDrone dataset, and detect 9 object categories, including `person, bicycles, car, van, truck, tricyle, awning-tricyle, bus, motor`.


## Getting Started
### 1. Training
a. Optional: Collect `tools/sniper_params_stats.py` to get the parameters including image target sizes, valid box ratio ranges, the chip target size and the stride, and modify them in configs/datasets/sniper_coco_detection.yml.
```bash
python tools/sniper_params_stats.py FasterRCNN annotations/instances_train2017.json
```
b. Optional: trian the detector to get negative proposals
```bash
python -m paddle.distributed.launch --log_dir=./sniper/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/sniper/faster_rcnn_r50_fpn_1x_sniper_visdrone.yml --save_proposals --proposals_path=./proposals.json &>sniper.log 2>&1 &
```
c. Train models
```bash
python -m paddle.distributed.launch --log_dir=./sniper/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/sniper/faster_rcnn_r50_fpn_1x_sniper_visdrone.yml --eval &>sniper.log 2>&1 &
```

### 2. Evaluation
Evaluate the result of models on COCO val2017 dataset with single GPU:
```bash
# Use the checkpoint saved in the training
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c configs/sniper/faster_rcnn_r50_fpn_1x_sniper_visdrone.yml -o weights=output/faster_rcnn_r50_fpn_1x_sniper_visdrone/model_final
```

### 3. Inference
Run the following commands to infer images with single GPU, and designate the image path through `--infer_img`, or designate the directory and infer all of its images through `--infer_dir`.

```bash
# Infer one single image
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/sniper/faster_rcnn_r50_fpn_1x_sniper_visdrone.yml -o weights=output/faster_rcnn_r50_fpn_1x_sniper_visdrone/model_final --infer_img=demo/P0861__1.0__1154___824.png

# Infer all the images in the directory
CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c configs/sniper/faster_rcnn_r50_fpn_1x_sniper_visdrone.yml -o weights=output/faster_rcnn_r50_fpn_1x_sniper_visdrone/model_final --infer_dir=demo
```

## Citations
```
@misc{1805.09300,
Author = {Bharat Singh and Mahyar Najibi and Larry S. Davis},
Title = {SNIPER: Efficient Multi-Scale Training},
Year = {2018},
Eprint = {arXiv:1805.09300},
}

@ARTICLE{9573394,
  author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and Fan, Heng and Hu, Qinghua and Ling, Haibin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Detection and Tracking Meet Drones Challenge},
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TPAMI.2021.3119563}}
```
