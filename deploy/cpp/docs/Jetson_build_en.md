[简体中文](Jetson_build.md)
# Compilation Guide of Jetson


## Explanation
NVIDIA Jetson is an embeded device with NVIDIA GPU, which can deploy object detection algorithms to the device. 
This doc is a tutorial of the deployment of `PaddleDetection` models on Jetson.

The demo here is based on Jetson TX2 and JetPack 4.3.

For the tutorial of developing Jetson, please refer to [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html).

## Setup of Jetson
For the installation of `Jetson` software, refer to [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html).

* (1) Check the version of 14t of the hardware system
```
cat /etc/nv_tegra_release
```
* (2) Choose an available version of `JetPack` considering the hardware. For their relationship, refer to [jetpack-archive](https://developer.nvidia.com/embedded/jetpack-archive).

* (3) Download `JetPack`. You can refer to `Preparing a Jetson Developer Kit for Use` in [NVIDIA Jetson Linux Developer Guide](https://docs.nvidia.com/jetson/l4t/index.html) to flash the mirror of the system.

**Note**: Choose a compatible `JetPack` version on [jetpack-archive](https://developer.nvidia.com/embedded/jetpack-archive) and then root.

## Download or Compile the `Paddle` Inference Library
本文档使用`Paddle`在`JetPack4.3`上预先编译好的预测库，请根据硬件在[安装与编译 Linux 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/05_inference_deployment/inference/build_and_install_lib_cn.html) 中选择对应版本的`Paddle`预测库。

这里选择[nv_jetson_cuda10_cudnn7.6_trt6(jetpack4.3)](https://paddle-inference-lib.bj.bcebos.com/2.0.0-nv-jetson-jetpack4.3-all/paddle_inference.tgz), `Paddle`版本`2.0.0-rc0`,`CUDA`版本`10.0`,`CUDNN`版本`7.6`，`TensorRT`版本`6`。

若需要自己在`Jetson`平台上自定义编译`Paddle`库，请参考文档[安装与编译 Linux 预测库](https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html) 的`NVIDIA Jetson嵌入式硬件预测库源码编译`部分内容。
