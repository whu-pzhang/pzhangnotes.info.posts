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

最近项目需要，要做目标跟踪以及轨迹预测的算法，翻了翻论文，应用比较多的就是卡尔曼滤波了。以前本科的时候就听过卡尔曼滤波，学大地测量的好像经常会用到这个，不过那时本身用不到，也就没关注。现在需要用到了，就去学习了一下，这里算是将其总结记录一下吧。

[Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter) 的提出到现在已经快60年了，但其



Kalman滤波应用广泛，只要是对动态系统不确定的状态估计，基本都可以看到kalman滤波的身影。



## 假设前提





## 状态预测

已知物体某一时刻的状态（例如：位置，速度，运动方向等），如何对其下一时刻的位置进行预测？卡尔曼滤波便是用来解决这一类问题而提出的。假设一个物体的状态向量表示如下：
$$
\vec{x} = \begin{bmatrix}
p \\
v
\end{bmatrix}
$$
卡尔曼滤波假设状态向量中的变量均符合高斯分布，具有均值$\mu$ ，以及方差 $\sigma^2$ 。 状态向量的协方差矩阵为 $\mathbf{P}$ ，表示状态向量内部变量之间的相关性以及自身的不确定性。设当前时刻为 $k-1$，那么下一时刻的状态向量可以用矩阵形式表示如下：
$$
\begin{align} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \begin{bmatrix} 
1 & \Delta t \\ 
0 & 1 
\end{bmatrix} \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
&= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}}
\end{align}
$$
$\mathbf{F}_k$ 称为状态转移矩阵（state transition matrix），也即是预测矩阵。通过该矩阵可以得到下个时刻的状态。那么状态向量的协方差呢？也通过该矩阵进行传递到下一状态。首先易知有如下关系：
$$
\begin{split}
Cov(x) &= \Sigma \\
Cov(\mathbf{A}x) &= \mathbf{A} \Sigma \mathbf{A}^T
\end{split}
$$
那么可得状态向量及其协方差矩阵更新如下：
$$
\begin{split} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\ 
\color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \mathbf{F}_k^T 
\end{split}
$$
虽然该方程可以预测出物体下个时刻的状态，但它并不完全。可能还有与状态向量无关的因素带来的影响。例如，汽车运动过程中对刹车或油门的控制。假设小车运动时的加速度为 $a$ ，那么我们的运动方程就变为：
$$
\begin{split}
p_k & = p_{k-1} + \Delta t & v_{k-1} + \frac{1}{2} a {\Delta t}^2 \\
v_k & =  & v_{k-1} + a \Delta t
\end{split}
$$
表示为矩阵形式：
$$
\begin{split}
\color{deeppink} {\mathbf{\hat{x}}_k} & = \mathbf{F}_k \color{royalblue} {\mathbf {\hat{x}}_{k-1}} + \begin{bmatrix}
\frac{{\Delta t}^2}{2} \\
\Delta t
\end{bmatrix} a \\

& = \mathbf{F}_{k} \color{royalblue} {\mathbf {\hat{x}}_{k-1}} + \mathbf{B}_k \color{darkorange}{\mathbf{u}_k}
\end{split}
$$


$\mathbf{B}_k$ 为控制矩阵（control matrix），$\mathbf{u}_k$ 为控制向量。 

到目前为止，若我们考虑了所有对运动状态有影响的因素，便可以根据上述式子对物体的运动状态进行递推。但实际上，还有一些外部的影响是我们没法控制的，比如四轴飞行器被风吹的摇晃，小车轮子打滑等。这些也都会对运动状态产生影响。在卡尔曼滤波中，我们假设这些不确定的因素为服从高斯分布的噪声，其协方差矩阵表示为$\mathbf{Q}_{k}$ 。如此一来，我们的递推关系式变为了：
$$
\begin{split} 
\color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} + \mathbf{B}_k \color{darkorange}{\mathbf{u}_k} \\ 
\color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \mathbf{F}_k^T + \color{mediumaquamarine} {\mathbf{Q}_k}
\end{split}
$$
状态 <font color=deeppink>**新的最优估计**</font> 是在 <font color=royalblue>**前一个最优估计**</font> 的基础预测出来的，再加上一个 <font color=darkorange>**已知的外部影响**</font> 的修正量。而<font color=deeppink>**新的不确定性**</font> 则是由 <font color=royalblue>**旧的不确定性**</font> 预测而来，再加上额外的来自 <font color=mediumaquamarine>**环境的不确定性**</font>。



现在我们利用当前状态预测出了由 $\color{deeppink} {\hat{ \mathbf{x}}_k}$ 和 $\color{deeppink} {\mathbf{P}_k}$ 所描述的系统下一时刻的状态，那么当我们有了来自传感器的实际观测值后，应当如何对系统进行更新呢？这里涉及到预测状态和观测状态的融合。

## 利用测量更新预测

传感器得到的测量值可能与状态向量单位以及标度不同，因此需要将状态向量转换为我们读取的测量值，转换矩阵为 $\mathbf{H}_k​$ 。 



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

类似的，量测（mesurement）方程如下：
$$
\boldsymbol{z}_t = \boldsymbol{H}_t \boldsymbol{x}_t + \boldsymbol{v}_t
$$
其中，

- $\boldsymbol{z}_t$ 为观测值
- $\boldsymbol{H}_t$ 为转换矩阵，将状态向量映射到测量值所在的空间上
- $\boldsymbol{v}_t$ 为测量的高斯噪声，均值为零，协方差矩阵为 $\boldsymbol{R}_t$



对目标运动状态向量的估计即为预测：
$$
\boldsymbol{\hat x}_{t|t-1} = \boldsymbol{F}_t  \boldsymbol{\hat x}_{t-1} + \boldsymbol{B}_t \boldsymbol{u}_t
$$
这一步是根据模型计算得到的，不含噪声。

$\boldsymbol{x}_t$ 为 $t$ 时刻状态向量的真值（不可知），那么预测误差 $\boldsymbol{e}_t$ 为：
$$
\boldsymbol{e}_t = \boldsymbol{x}_t - \boldsymbol{\hat x}_{t|t-1} = \boldsymbol{F}_t \left( \boldsymbol{x}_{t-1} - \boldsymbol{\hat x}_{t-1} \right) + \boldsymbol{w}_{t}
$$
误差的协方差矩阵 $\boldsymbol{P}_{t|t-1}$ 为：
$$
\begin{align}
\boldsymbol{P}_{t|t-1} &= E \left( \boldsymbol{e}_t \boldsymbol{e}_t^T \right) \\
  &= E \left[ \left( \boldsymbol{F}_t \left( \boldsymbol{x}_{t-1} - \boldsymbol{\hat x}_{t-1} \right) + \boldsymbol{w}_{t} \right) \cdot \left( \boldsymbol{F}_t \left( \boldsymbol{x}_{t-1} - \boldsymbol{\hat x}_{t-1} \right) + \boldsymbol{w}_{t} \right)^T \right]
\end{align}
$$


## 附录
$$
\begin{split}
Cov(x) &= \Sigma \\
Cov(\mathbf{A}x) &= \mathbf{A} \Sigma \mathbf{A}^T
\end{split}
$$



## 参考

- [How a kalman filter works in pictures](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
- [Kalman filter](https://en.wikipedia.org/wiki/Kalman_filter)
- 