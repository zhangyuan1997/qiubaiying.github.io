---
layout:     post
title:      Win10下的TensorFlow_gpu-1.12.0安装
subtitle:   Install
date:       2019-01-25
author:     Xiao Zhang
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - TensorFlow_GPU
	- CUDA
	- CUDNN
---

转载自[https://blog.csdn.net/Fowee/article/details/84983245](https://blog.csdn.net/Fowee/article/details/84983245)

基础环境：Win10、Python3.6.7、Pycharm2018 2.4

上述安装：https://blog.csdn.net/Fowee/article/details/83048154

---

安装环境：

显卡：GTX1060(Notebook)

CPU：i7-8750H

安装目标：

tensorflow_gpu-1.12.0 from: tensorflow.google.cn/install/

其他需求：

CUDA：10.0.130_411.31

cuDNN：7.3.1.20

Compiler：VS2017 15.8 (本文未涉及，如果没有自行下载)

版本参照：

https://github.com/fo40225/tensorflow-windows-wheel

https://tensorflow.google.cn/install/source_windows

本文涉及的所有下载文件：

链接：https://pan.baidu.com/s/1L3UtuYSyc2esf_W0bZjVmw 

提取码：j36w

---
本文结构：

1.CUDA

（1）显卡型号检查

（2）CUDA下载

（3）CUDA安装

（4）CUDA测试

2.cuDNN

（1）cuDNN下载

（2）配置环境变量

3.TensorFlow

（1）下载对应的GPU版本的TF

（2）安装TF

（3）Hello TensorFlow！

---

## 1.CUDA 10.0.130_411.31

CUDA（Compute Unified Device Architecture），是显卡厂商NVIDIA推出的运算平台。 CUDA™是一种由NVIDIA推出的通用并行计算架构，该架构使GPU能够解决复杂的计算问题。 它包含了CUDA指令集架构（ISA）以及GPU内部的并行计算引擎[百度百科]。

（1）显卡型号检查：https://developer.nvidia.com/cuda-gpus

特别说明，如果Compute Capability值低于3.0，不建议使用该GPU计算。

![image](https://img-blog.csdnimg.cn/20181213102526546.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

（2）CUDA下载

文档：https://developer.nvidia.com/cuda-toolkit-archive

![image](https://img-blog.csdnimg.cn/20181213105052740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

下载CUDA：https://developer.nvidia.com/cuda-downloads

根据 https://github.com/fo40225/tensorflow-windows-wheel  TF1.11-1.12 必须安装CUDA10

![image](https://img-blog.csdnimg.cn/2018121310491392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

（3）CUDA安装

下载后双击打开安装，

![image](https://img-blog.csdnimg.cn/20181213122948416.png)

这里路径修改为：D:\NVIDIA\CUDA  （之后的cuDNN也应该在该路径下）

![image](https://img-blog.csdnimg.cn/2018121311273355.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

等待一会，选择自定义

![image](https://img-blog.csdnimg.cn/20181213123236888.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

这里取消勾选GeForce Experience，保留其他组件

![image](https://img-blog.csdnimg.cn/20181213123504985.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

修改安装路径，这里分别是：

D:\NVIDIA\CUDAv10.0

D:\NVIDIA\CUDAv10.0

D:\NVIDIA\Samplesv10.0

![image](https://img-blog.csdnimg.cn/20181213123934492.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

同意并安装：

![image](https://img-blog.csdnimg.cn/20181213124105573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

耐心等待安装完成。

（4）CUDA测试

在 CMD 中输入 nvcc -V，如果输出是 CUDA 版本信息，则说明安装成功

![image](https://img-blog.csdnimg.cn/20181213132959453.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

## 2.cuDNN 7.3.1.20

cuDNN官网：https://developer.nvidia.com/cudnn
（下载需先注册登录，由于很简单这里不赘述）

单击下载：cudnn-10.0-windows10-x64-v7.3.1.20

![image](https://img-blog.csdnimg.cn/20181213132201337.png)

解压缩下载的cudnn-10.0-windows10-x64-v7.3.1.20.zip文件，得到3个文件夹：bin，include，lib

![image](https://img-blog.csdnimg.cn/20181213132231172.png)

将这三个文件拷贝到“D:\NVIDIA\CUDAv10.0” （CUDA安装路径）

![image](https://img-blog.csdnimg.cn/2018121313232638.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

（2）配置环境变量

确认环境变量，CUDA_PATH和CUDA_PATH_V10已经存在

![image](https://img-blog.csdnimg.cn/20181213132451525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

依此添加下列三个路径到Path里


D:\NVIDIA\CUDAv10.0\bin
 
D:\NVIDIA\CUDAv10.0\include
 
D:\NVIDIA\CUDAv10.0\lib\x64

![image](https://img-blog.csdnimg.cn/2018121313280667.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

## 3.TensorFlow

（1）下载对应的GPU版本的TF

我们选择AVX2指令集版本，请先查看自己CPU是否支持AVX2指令集，可以通过下载CPU-Z检测：

![image](https://img-blog.csdnimg.cn/20181213111846296.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

https://github.com/fo40225/tensorflow-windows-wheel/tree/master/1.12.0/py36/GPU/cuda100cudnn73avx2
路径下分别下载001和002（或直接点击001、002分别下载）

如果不支持AVX指令集，请下载 SSE2指令集版本：TensorFlow1.12.0 win10 GPU SSE2

下载完成后，将文件解压，

![image](https://img-blog.csdnimg.cn/20181213112103492.png)

解压后得到：

![image](https://img-blog.csdnimg.cn/2018121311232858.png)

（2）安装TF

CMD下执行命令：

进入 .whl 文件的目录

cd /d E:\Downloads\tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64

通过pip命令安装

pip install tensorflow_gpu-1.12.0-cp36-cp36m-win_amd64.whl

![image](https://img-blog.csdnimg.cn/20181213134653457.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

耐心等待安装完成

![image](https://img-blog.csdnimg.cn/2018121313473258.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

（3）Hello TensorFlow！

进入Python，依此输入


```
import tensorflow as tf
 
hello = tf.constant("Hello,TensorFlow！")
sess = tf.Session()
 
print(sess.run(hello))
 
sess.close()
```


![image](https://img-blog.csdnimg.cn/20181213140753744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)

成功打印：Hello TensorFlow！

附Pycharm中的运行结果：

![image](https://img-blog.csdnimg.cn/20181213141225904.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0Zvd2Vl,size_16,color_FFFFFF,t_70)