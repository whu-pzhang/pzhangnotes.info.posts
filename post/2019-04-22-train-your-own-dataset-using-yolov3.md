---
title: YOLOv3 笔记
author: pzhang
date: 2019-04-22
mathjax: true
categories:
  - 计算机视觉
  - 深度学习
tags:
  - Python
  - NumPy
  - C

draft: true
slug: train-your-own-dataset-using-yolov3
---


## 前言

YOLO是最受大家欢迎的实时目标检测框架之一。

本文将介绍如何利用自己的数据来训练一个目标检测器，主要介绍一些注意事项以及一些小的trick。

<!--more-->

## 数据集

这里以OpenImage中的数据集为例。

### 数据下载

### 划分测试集


## Darknet


### 配置文件

YOLOv3从配置文件中读取网络结构以及学习率这些超参数。

```bash
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64   
subdivisions=16
```

`batch` 为训练时batch size的大小，可根据自己显存大小进行设置。

`subdivisions` 表示将batch size分为多少份进行处理，同样也是为了应对显存不足的情况。
`batch` 通常设置为64，`subdivisions` 的值可以从1开始指定，若提示 `out of memory`，则可以增加该参数的值，如2，4，8，16等，直到可以正常训练。

```bash
width=608
height=608
channels=3
```

这三项为指定网络输入图像的大小和通道数。但实际的图像尺寸不必和该参数保持一致。训练和推断时，
yolo会保持宽高比例的将输入图像resize为该参数指定的大小。


**多GPU训练时，需根据GPU数对参数进行相应的调整：**

``` bash
learning_rate = learning_rate / num_gpus
burn_in = burn_in * num_gpus

```