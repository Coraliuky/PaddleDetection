[简体中文](README.md)| English
# Inference and Deployment using C++



## Compiling and Deployment Tutorials in Different Environments
- [Linux](docs/linux_build.md)
- [Windows](docs/windows_vs2019_build.md) (Visual Studio 2019 is used)
- [NV Jetson](docs/Jetson_build.md)

## Overview of Deployment using C++
1. [Explanation](#1Explanation)
2. [Directory and Docs](#2Directory and Docs)


### 1. Explanation

The directory provides users with a cross-platform deployment project using C++. Users can export models trained by 'PaddleDetection' and run them based on this project. They can also quickly integrated code into their own practical projects.

The project is designed to be:
- Cross-platform: helping in compiling, integration of second development, and deployment in `Windows` or `Linux`
- Extensible: making it available for users to develop customized logics like data pre-processing ones based on new models
- High-performance: optimizing performance in key steps for characteristics of image detection basides the intrinsic advantages of 'PaddlePaddle'
- Compatible: supporting various structures of detection models, including `Yolov3`/`Faster_RCNN`/`SSD`

### 2. Directory and Docs

```bash
deploy/cpp
|
├── src
│   ├── main.cc # example of integration code, program entry
│   ├── object_detector.cc # model loading and encapsulation of main inference logic 
│   └── preprocess_op.cc # encapsulation of main logic related to pre-processing
|
├── include
│   ├── config_parser.h # export yaml file parsing of model config
│   ├── object_detector.h # model loading and encapsulation of main inference logic
│   └── preprocess_op.h # encapsulation of main logic related to pre-processing
|
├── docs
│   ├── linux_build.md # Linux compilation guide
│   └── windows_vs2019_build.md # compilation guide of Windows VS2019
│
├── build.sh # compilation script
│
├── CMakeList.txt # cmake compilation entry file
|
├── CMakeSettings.json # Visual Studio 2019 CMake compilation setting
└── cmake # dependent external projects cmake（only `yaml-cpp` currently）

```
