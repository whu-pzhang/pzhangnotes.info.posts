---
title: 卡尔曼滤波
author: pzhang
date: 2018-09-27T22:20:58+08:00
lastMod: 2018-09-27T22:20:58+08:00
categories:
  - 计算机视觉
tags:
  - 理论

draft: true
slug: kalman-filter
---

## 引言

最近项目需要，要做目标跟踪以及轨迹预测的算法，翻了翻论文，应用比较多的就是卡尔曼滤波了。以前本科的时候就听过卡尔曼滤波，学大地测量GPS的好像经常会用到这个，不过那时本身用不到，也就没关注。现在需要用到了，就去学习了一下，这里算是将其总结记录一下吧。

[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) 的提出到现在已经快60年了，但其



## 问题的提出

已知物体某一时刻的状态（例如：位置，速度，运动方向等），如何对其下一时刻的位置进行预测？

卡尔曼滤波模型中通过前一个时刻的状态来预测目标当前时刻的状态
$$
\boldsymbol{x}_t = \boldsymbol{F}_t  \boldsymbol{x}_{t-1} + \boldsymbol{B}_t \boldsymbol{u}_t + \boldsymbol{w}_t
$$

- $\boldsymbol{x}_t$ 为 $t$ 时刻的状态向量，如目标位置，速度等
- $\boldsymbol{u}_t$ 为运动测量值，如加速度，转向等
- $\boldsymbol{F}_t$ 为状态转移矩阵，将 $t-1$ 时刻的状态传递到 $t$ 时刻
- $\boldsymbol{B}_t$ 为控制输入矩阵，将运动量测值 $\boldsymbol{u}_t$ 的作用映射到状态向量上去
- $\boldsymbol{w}_t$ 是状态向量中每一项的噪声，服从均值为零，协方差矩阵为 $\boldsymbol{Q}_t$ 的高斯分布

这个方程为状态方程，即卡尔曼滤波中的预测（predict）过程。

类似的，量测（mesurament）方程如下：
$$
\boldsymbol{z}_t = \boldsymbol{H}_t \boldsymbol{x}_t + \boldsymbol{v}_t
$$
其中，

- $\boldsymbol{z}_t$ 为观测值
- $\boldsymbol{H}_t$ 为转换矩阵，将状态向量映射到测量值所在的空间上
- $\boldsymbol{v}_t$ 为测量的高斯噪声，均值为零，协方差矩阵为 $\boldsymbol{R}_t$  

