---
title: cs231n作业2笔记
author: pzhang
date: 2018-11-10
lastMod: 2018-11-10
markup: mmark
mathjax: true
categories:
  - 计算机视觉
tags:
  - Python
  - NumPy

draft: true
slug: cs231n-assignment2
---

## 简介

Assignment1中我们分别采用 `kNN`、线性分类器、`SVM`、`Softmax`分类以及简单两层神经网络对CIFAR-10数据集进行了分类。

来到Assignment2中，需要自己搭建卷积神经网络（CNN）来进行图像分类。

Assignment2主要分为5个部分：

- **Fully-connected Nerual Network**

这部分为作业1中两层神经网络的延续，需要将各个隐藏层的`forward`以及`backward`过程模块化，以便搭建任意层数的神经网络。

- **Batch Normalization**

为了使深度神经网络的训练更快，一种策略是寻找更好的优化方法；另一种思路便是改变网络的架构使其训练更快。我们知道，机器学习方法对于白化（特征之间无相关性，零均值，单位方差）的输入数据有着更好的效果。

对数据的预处理（去均值、归一化、PCA以及白化）只能保证第一层的输入满足条件，但随着网络的逐步深入，深层网络的激活输入就不再满足条件了。更糟的是，随着训练加深，每层的特征分布还会随着权重的更新而偏移。

[loffe 2015](https://arxiv.org/abs/1502.03167)提出的Batch Normalization便是为了解决这个问题的。具体可以参考论文。

- **Dropout**

Dropout 是缓解过拟合的一种方法。会按比例随机的丢掉一部分激活值，达到缓解过拟合的目的。


- **Convolutional Networks**

这部分需要完成CNN的各个组件，然后组合在一起，搭建一个完整的CNN网络。


- **PyTorch/TensofFlow on CIFAR-10**

主要是这两个框架的练习。


我的代码实现[whu-pzhang/cs231n](https://github.com/whu-pzhang/cs231n).

<!--more-->

## Fully-connected Nerual Network

这部分需要对每层实现一个 `forward` 和 `backward` 函数：`forward` 函数接收为输入、权重和其他参数，返回输出以及一个 `cache` 对象，包含反向传播计算梯度时用到的数据。

```python
def layer_forward(x, w):
  """ Receive inputs x and weights w """
  # Do some computations ...
  z = # ... some intermediate value
  # Do some more computations ...
  out = # the output

  cache = (x, w, z, out) # Values we need to compute gradients

  return out, cache
```

`backward` 函数则接收upstream传过来的梯度和 `forward` 返回的 `cache` 对象，
计算返回该层相对于输入和权重的梯度值。

```python
def layer_backward(dout, cache):
  """
  Receive dout (derivative of loss with respect to outputs) and cache,
  and compute derivative with respect to inputs.
  """
  # Unpack cache values
  x, w, z, out = cache

  # Use values in cache to compute derivatives
  dx = # Derivative of loss with respect to x
  dw = # Derivative of loss with respect to w

  return dx, dw
```

### affine layer

全连接层的forward就是简单的矩阵乘法。需要先将每个实例展平为向量，然后和权重做矩阵乘法即可，最后别忘了偏置项。

```python
x_reshaped = x.reshape(x.shape[0], -1)
out = x_reshaped.dot(w) + b
```

backward时，根据链式法计算即可，注意各个量的shape即可。

```python
dx = dout.dot(w.T).reshape(*x.shape)
dw = x.reshape(x.shape[0], -1).T.dot(dout)
db = np.sum(dout, axis=0)
```

### ReLU activation

`ReLU` 激活只是应用一个mask，计算相当简单，直接用 `np.maximum()` 函数即可。

backward时，注意只有大于零的项有梯度值。

### 组合层

将affine层和ReLU组合起来即可

### loss layers

在作业1中实现的`Softmax`和`SVM`损失函数可以直接拿过来用，不再赘述。

### 搭建多层神经网络

现在直接将前面实现的层组合起来便可以实现全连接层神经网络。
搭建完成后，利用提供的`Solver`类，便可以实现神经网络的训练和验证。

### 优化方法

根据讲义里的各种优化方法，直接实现即可，没啥难度。主要是理解每个优化方法的思想。
依次是 `SGD+Momentum`，`RMSProp` 和 `Adam`。


## Batch Normalization

Batch Normalization是为了克服层数较多的神经网络在训练时的 internal covariate shift 现象，减弱梯度饱和。此外，还可以加快训练速度，学习率也可以设置的较大。

### forward

设BN层的小批量输入为 $\boldsymbol{X} \in \mathbb{R}^{ N \times D}$。
在forward过程中，首先计算每个特征的均值和方差：

$$
\begin{align}
\boldsymbol{\mu} &= \frac{1}{N} \sum_{k=1}^N \boldsymbol{x}_k \\
\boldsymbol{\sigma}^2 & = \frac{1}{N} \sum_{k=1}^N { (\boldsymbol{x}_k - \boldsymbol{\mu})^2 }
\end{align}
$$

接着就可以对输入进行归一化：

$$
\boldsymbol{\hat{x}}_i = \frac{ \boldsymbol{x}_i - \boldsymbol{\mu}} { \sqrt{\boldsymbol{\sigma}^2 + \epsilon} }
$$

再进行平移和缩放：

$$
\boldsymbol{y}_i = \gamma \, \boldsymbol{\hat x}_i + \beta
$$

其中，$\gamma, \beta \in \mathbb{R}^{1 \times D}$。整个forward过程可记为：

$$
\boldsymbol{\hat{X}} = \frac{\boldsymbol{X} - \boldsymbol{\mu}} {\sqrt{ \boldsymbol{\sigma}^2 + \epsilon}} \\
\boldsymbol{Y} = \gamma \odot \boldsymbol{\hat{X}} + \beta
$$

至此，正向传播就完成了。

为了方便计算梯度的反向传播，根据计算图可将`forward`过程分解为如下9步：

```python
mu = 1.0 / N * np.sum(x, axis=0, keepdims=True)               # (1)
xsubmu = x - mu                                               # (2)
xsubmusqr = xsubmu**2                                         # (3)
var = 1.0 / N * np.sum(xsubmusqr, axis=0, keepdims=True)      # (4)
sqrtvar = np.sqrt(var + eps)                                  # (5)
invsqrtvar = 1.0 / sqrtvar                                    # (6)
x_norm = xsubmu * invsqrtvar                                  # (7)
gammax = gamma * x_norm                                       # (8)
out = gammax + beta                                           # (9)
```

### backward

`backward` 过程按照上面9步倒序进行即可：

```python
dgammax = dout                                                # (9)
dbeta = np.sum(dout, axis=0, keepdims=True)                   # (9)
dgamma = np.sum(dgammax * x_norm, axis=0, keepdims=True)      # (8)
dx_norm = dgammax * gamma                                     # (8)
dxsubmu = dx_norm * invsqrtvar                                # (7)
dinvsqrtvar = np.sum(dx_norm * xsubmu, axis=0, keepdims=True) # (7)
dsqrtvar = dinvsqrtvar * (-1.0) * (sqrtvar)**(-2)             # (6)
dvar = dsqrtvar * (0.5 * (var + eps)**(-0.5))                 # (5)
dxsubmusqr = dvar * (1.0 / N * np.ones((N, D)))               # (4)
dxsubmu += dxsubmusqr * (2 * xsubmu)                          # (3)
dx = dxsubmu                                                  # (2)
dmu = -1.0 * np.sum(dxsubmu, axis=0, keepdims=True)           # (2)
dx += dmu * (1.0 / N * np.ones((N, D)))                       # (1)
```

### Alternative backward implement

BatchNorm backward前面的实现是按照计算图一步步来的，实际上可以直接推导出BatchNorm的梯度计算公式，这样能够简化计算。

设 $L$ 为训练损失，我们已知从upstream传来的 $\frac{\partial L}{\partial \boldsymbol{Y}} \in \mathbb{R}^{N \times D}$， 反向传播时，我们需要计算：

1. $\frac{\partial L}{\partial \beta} \in \mathbb{R}^{1 \times D}$
2. $\frac{\partial L}{\partial \gamma} \in \mathbb{R}^{1 \times D}$
3. $\frac{\partial L}{\partial \boldsymbol{X}} \in \mathbb{R}^{N \times D}$。

$\frac{\partial L}{\partial \beta}$ 和 $\frac{\partial L}{\partial \gamma}$ 的计算很直观：

$$
\begin{align}
\frac{\partial L}{\partial \gamma} & = \sum_{i}^N { \frac{\partial L} {\partial \boldsymbol{y}_i} \odot \boldsymbol{\hat{x}_i}} \\
\frac{\partial L}{\partial \beta} & = \sum_i^N {\frac{\partial L} {\partial \boldsymbol{y}_i}}
\end{align}
$$

而 $\frac{\partial L}{\partial \boldsymbol{X}}$ 的计算则比较复杂。
由于 $\boldsymbol{\mu}, \boldsymbol{\sigma}$ 都是 $\boldsymbol{X}$ 的函数，根据链式法则：

$$
\frac{\partial L}{\partial \boldsymbol{X}} =
\frac{\partial L}{\partial \boldsymbol{\hat{X}}} \frac{\partial \boldsymbol{\hat{X}}}{\partial \boldsymbol{X}} +
\frac{\partial L}{\partial \boldsymbol{\sigma}^2} \frac{\partial \boldsymbol{\sigma}^2}{\partial \boldsymbol{X}} +
\frac{\partial L}{\partial \boldsymbol{\mu}} \frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{X}}
$$

我们可以依次计算这三项。第一项比较简单：

$$
\begin{align}
\frac{\partial L}{\partial \boldsymbol{\hat{X}}} & =
\gamma \odot \frac{\partial L}{\partial \boldsymbol{Y}} \\
\frac{\partial \boldsymbol{\hat{X}}}{\partial \boldsymbol{X}} & =
(\boldsymbol{\sigma}^2 + \epsilon) ^ {-1/2} \\
 & \Downarrow \\
\frac{\partial L}{\partial \boldsymbol{\hat{X}}} \frac{\partial \boldsymbol{\hat{X}}}{\partial \boldsymbol{X}} & =
\gamma \odot \frac{\partial L}{\partial \boldsymbol{Y}} \, (\boldsymbol{\sigma}^2 + \epsilon) ^ {-1/2}
\end{align}
$$

接下来计算第二项：

$$
\frac{\partial L}{\partial \boldsymbol{\sigma}^2} =
\frac{\partial L}{\partial \boldsymbol{\hat{X}}} \,
\frac{\partial \boldsymbol{\hat{X}}}{\partial \boldsymbol{\sigma}^2} =
-\frac{\gamma (\boldsymbol{\sigma}^2 + \epsilon)^{-3/2}}{2} \,
\sum_{i}^{N} {\frac{\partial L}{\partial \boldsymbol{y}_i} (\boldsymbol{x}_i - \boldsymbol{\mu}) }
$$

计算相对于 $\boldsymbol{\sigma}$ 的梯度时，需要沿着批量中所有的实例求和，对 $\boldsymbol{\mu}$ 求导（第三项）时也是一样：

$$
\begin{align}
\frac{\partial L}{\partial \boldsymbol{\mu}} & =
\frac{\partial L}{\partial \boldsymbol{\hat{X}}}
\frac{\partial \boldsymbol{\hat{X}}}{\partial \boldsymbol{\mu}} +
\frac{\partial L}{\partial \boldsymbol{\sigma}^2}
\frac{\partial \boldsymbol{\sigma}^2}{\partial \boldsymbol{\mu}} \\
& = -\gamma (\boldsymbol{\sigma}^2 + \epsilon)^{-1/2} \,
\sum_{i}^{N} {\frac{\partial L}{\partial \boldsymbol{y}_i}} +
\frac{\partial L}{\partial \boldsymbol{\sigma}^2} \cdot (-\frac{2}{N}) \cdot \sum_i^N (\boldsymbol{x}_i - \boldsymbol{\mu}) \\
& = -\gamma (\boldsymbol{\sigma}^2 + \epsilon)^{-1/2} \,
\sum_{i}^{N} {\frac{\partial L}{\partial \boldsymbol{y}_i}}
\end{align}
$$

接下来分别求得 $\frac{\partial \boldsymbol{\sigma}^2}{\partial \boldsymbol{X}}$ 和 $\frac{\partial \boldsymbol{\mu}}{\partial \boldsymbol{X}}$ 即可。

### Layer Normalization

Layer Normalization 是沿着另外一个轴进行的 Batch Normalization

## Convolutional Networks

卷积神经网络（CNN）其实和常规的神经网络很像，由包含可学习的权重和偏置的神经元组成。
每个神经元参数与输入做点积得到新的输出。CNN中通常包括卷积层（Convolutional layer）、池化层（Pooling layer）和全联接层（Fully-connected layer）。

卷积层是CNN的核心。在处理图像这类高维的输入时，不可能将当前神经元与输入的全部神经元连接起来。在卷积层中，只将当前神经元与输入数据的局部区域进行连接。即神经元的局部感受野（Local receptive field），也即是卷积核的大小，这是一个Hyperparameter。此外，主要注意的是，这种空间局部连接在深度轴方向总是与输入的深度相等的。

卷积层的输出大小由三个Hyperparameter控制：

1. 输出深度（depth）

对应着卷积层中卷积核的个数。每个卷积核在训练过程中会从输入中提取出不同的特征。

2. 滑动步长（stride）

控制着卷积核每次移动的像素数。当stride为2时，卷积核每次移动两个像素。

3. zero-padding

输入数据边缘需要填充的大小。可以控制输出的空间大小。

输出的空间大小是着三个参数的函数。设输入数据大小为 $W$，卷积层神经元感受野大小为$F$，卷积核移动步幅为$S$，zero-padding大小为$P$，那么该卷积层输出的大小为

$$
(W - F + 2P) / S + 1
$$

卷积层的另外一个重要特点是参数共享。

### forward

先来看单通道，`S=1，P=0`的最简单的情况，设输入$X$ 为 $3 \times 3$，卷积核$K$为 $2 \times 2$，那么输出$Y$的大小为 $(3 - 2)/1 + 1 = 2$

$$
\begin{pmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{33} \\
x_{31} & x_{32} & x_{33}
\end{pmatrix} \otimes
\begin{pmatrix}
k_{11} & k_{12} \\
k_{21} & k_{22}
\end{pmatrix} =
\begin{pmatrix}
y_{11} & y_{12} \\
y_{21} & y_{22}
\end{pmatrix}
$$

即

<!-- $$
\begin{align}
\begin{pmatrix}
y_{11} \\
y_{12} \\
y_{21} \\
y_{22}
\end{pmatrix} & =
\begin{pmatrix}
x_{11} k_{11} + x_{12} k_{12} + x_{21} k_{21} + x_{22} k_{22} \\
x_{12} k_{11} + x_{13} k_{12} + x_{22} k_{21} + x_{23} k_{22} \\
x_{21} k_{11} + x_{22} k_{12} + x_{31} k_{21} + x_{32} k_{22} \\
x_{22} k_{11} + x_{23} k_{12} + x_{32} k_{21} + x_{33} k_{22}
\end{pmatrix} \\
& =
\begin{pmatrix}
x_{11} & x_{12} & x_{21} & x_{22} \\
x_{12} & x_{13} & x_{22} & x_{23} \\
x_{21} & x_{22} & x_{31} & x_{32} \\
x_{22} & x_{23} & x_{32} & x_{33}
\end{pmatrix}
\begin{pmatrix}
k_{11} \\
k_{12} \\
k_{21} \\
k_{22}
\end{pmatrix}
\end{align}
$$ -->

$$
\begin{align}
\begin{pmatrix}
y_{11} & y_{12} & y_{21} & y_{22}
\end{pmatrix} & =
\begin{pmatrix}
k_{11} & k_{12} & k_{21} & k_{22} \\
\end{pmatrix}
\begin{pmatrix}
x_{11} & x_{12} & x_{21} & x_{22} \\
x_{12} & x_{13} & x_{22} & x_{23} \\
x_{21} & x_{22} & x_{31} & x_{32} \\
x_{22} & x_{23} & x_{32} & x_{33}
\end{pmatrix}
\end{align}
$$

卷积最终转化为了矩阵乘的形式。转换之后的 $X$， $K$ 和 $Y$ 分别为 $XC$， $KC$ 和 $YC$。
$XC$的每一列为卷积核的局部区域（感受野）展平得到。$KC$ 和 $YC$ 则分别是将 $K$ 和 $Y$ 展平后得到的行向量。

对于更常见的多通道输入，也是一样的原理：先将每个卷积核对应的局部感受野展平为一个列向量
（该操作称为`im2col`），然后将卷积核展平为行向量。以AlexNet为例，输入图像形状为 $227 \times 227 \times 3$，
卷积核形状为 $11 \times 11 \times 3$，步幅为4。那么就将卷积核的每个感受野展平为
$11 \times 11 \times 3=363$ 大小的列向量。以步幅4遍历整个图像后，卷积核的输出宽和高均为 $(227-11)/4+1=55$，
展平的话就是大小$55 \times 55=3025$的行向量。那么一幅图像经 `im2col` 后就变换成了大小 $363 \times 3025$ 的矩阵。

卷积层的权重参数也是类似的展平为行向量，AlexNet中第一个卷积层深度为96，那么经展平后，权重就变为了
$96 \times 363$ 大小的矩阵。

### backward

反向传播时，需要求$\boldsymbol{x}$, $\boldsymbol{w}$ 和 偏置项 $\boldsymbol{b}$ 这三项的梯度。



根据链式法则，$\boldsymbol{x}$ 的梯度表示如下：

$$
\frac{\partial L} {\partial \boldsymbol{x}} = \frac{\partial L} {\partial \boldsymbol{y}} \frac{\partial \boldsymbol{y}} {\partial \boldsymbol{x}}
$$

其中，$\frac{\partial L} {\partial \boldsymbol{y}}$ 为后面（通常为池化层或激活函数）传过来的梯度，记为 $\boldsymbol{\delta}$.

通过前面forward过程，可将$\boldsymbol{x}$的梯度依次写出来：

$$
\begin{split}
\frac{\partial L} {\partial x_{11}} & =
\boldsymbol{\delta} \cdot
\begin{pmatrix}
k_{11} & 0 \\
0 & 0
\end{pmatrix} & =
\delta_{11} k_{11} \\
\frac{\partial L} {\partial x_{12}} & =
\boldsymbol{\delta} \cdot
\begin{pmatrix}
k_{12} & k_{11} \\
0 & 0
\end{pmatrix} & = \delta_{11} k_{12} + \delta_{12} k_{11} \\
\frac{\partial L} {\partial x_{13}} & =
\boldsymbol{\delta} \cdot
\begin{pmatrix}
0 & k_{12} \\
0 & 0
\end{pmatrix} & = \delta_{12} k_{12} \\
 & & \vdots \\
\frac{\partial L} {\partial x_{22}} & =
\boldsymbol{\delta} \cdot
\begin{pmatrix}
k_{22} & k_{21} \\
k_{12} & k_{11}
\end{pmatrix} & = \delta_{11} k_{22} + \delta_{12} k_{21} + \delta_{21} k_{12} + \delta_{22} k_{11} \\
 & & \vdots \\
\frac{\partial L} {\partial x_{33}} & =
\boldsymbol{\delta} \cdot
\begin{pmatrix}
0 & 0 \\
0 & k_{22}
\end{pmatrix} & =
\delta_{22} k_{22}
\end{split}
$$

### Max pooling

### Spatial batch normalization

BatchNorm 不仅可以加快全连接深度神经网络的训练过程，而且对CNN也有效，只不过需要略微的调整，调整后称为 Spatial batch normalization.

BatchNorm 是沿着minibatch维做归一化。假设

### Group normalization
