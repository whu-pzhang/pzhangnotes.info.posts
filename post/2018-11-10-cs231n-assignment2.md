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

这部分为作业1中两层神经网络的延续，进一步需要将各个隐藏层的`forward`以及`backward`过程模块化，以便搭建任意层数的神经网络。

- **Batch Normalization**

为了使深度神经网络的训练更快，一种策略是寻找更好的优化方法；另一种思路便是改变网络的架构使其训练更快。我们知道，机器学习方法对于白化（特征之间无相关性，零均值，单位方差）的输入数据有着更好的效果。

对数据的预处理（去均值、归一化、PCA以及白化）只能保证第一层的输入满足条件，但随着网络的逐步深入，深层网络的激活输入就不再满足条件了。更糟的是，随着训练加深，每层的特征分布还会随着权重的更新而偏移。

[loffe 2015](https://arxiv.org/abs/1502.03167)提出的Batch Normalization便是为了解决这个问题的。具体可以参考论文。

- **Dropout**


- **Convolutional Networks**


- **PyTorch/TensofFlow on CIFAR-10**


我的代码实现[whu-pzhang/cs231n](https://github.com/whu-pzhang/cs231n).

<!--more-->

## Fully-connected Nerual Network




## Batch Normalization


### forward
假设每个小批量的训练集大小为 $m \times d$, 那么对每个批量的数据进行标准归一化可表示如下：

$$
\begin{align}
\mu_{\mathcal{B}} &\leftarrow \frac{1} {m} \sum_{i=1}^m {x_i} \\
\sigma_{\mathcal{B}}^2 & \leftarrow \frac{1}{m} \sum_{i=1}^m { (x_i - \mu_{\mathcal{B}})^2 } \\
\hat{x}_i & \leftarrow \frac{ x_i - \mu_{\mathcal{B}}} {\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \\
y_i & \leftarrow \gamma \hat{x}_i + \beta
\end{align}
$$

现在来推导BN层的反向传播求导公式。上游梯度已知为 `dout`

设输入为：

$$
\mathbf{X} = \begin{bmatrix}
\boldsymbol{x}_1 & \boldsymbol{x}_2 & \cdots & \boldsymbol{x}_N
\end{bmatrix} ^T =
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1d} \\
x_{21} & x_{22} & \cdots & x_{2d} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{md}
\end{bmatrix}
$$

在forward过程中，首先计算均值和方差如下：


$$
\begin{align}
\boldsymbol{\mu} &= \frac{1}{N} \sum_{k=1}^N \boldsymbol{x}_k \\
\boldsymbol{\sigma}^2 & = \frac{1}{N} \sum_{k=1}^N { (\boldsymbol{x}_k - \boldsymbol{\mu})^2 }
\end{align}
$$

接着得到标准差 $\boldsymbol{\sigma} = \sqrt{\boldsymbol{\sigma}^2 + \epsilon}$就可以对输入进行归一化：

$$
\boldsymbol{\hat{x}}_i = \frac{ \boldsymbol{x}_i - \boldsymbol{\mu}} { \sqrt{\boldsymbol{\sigma}^2 + \epsilon} }
$$

再进行平移和缩放：

$$
\boldsymbol{y}_i = \gamma \, \boldsymbol{\hat x}_i + \beta
$$


至此，正向传播就完成了。

为了方便计算梯度的反向传播，将`forward`过程分解为如下9步：

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
```

反向传播时，我们需要计算 $\frac{\partial L}{\partial \boldsymbol{X}}$。

由链式法则，可知：

$$
\frac{\partial L}{\partial \boldsymbol{x}_i} = \frac{\partial L}{\partial \boldsymbol{y}} \cdot \frac{\partial \boldsymbol{y}} {\partial \boldsymbol{\hat{x}}} \cdot \frac{\partial \boldsymbol{\hat{x}}} {\partial \boldsymbol{x}_i}
$$

其中，$\frac{\partial L} {\partial \boldsymbol{\hat{x}}_i}$ 已由上游梯度给出，因此只需计算后半部分即可。这里设 $m=1, d=3$，则

$$
\begin{align}
\frac{\partial \boldsymbol{\hat{x}}}{\partial \boldsymbol{x}_i} &= \frac{\partial } {\partial \boldsymbol{x}_i} { (\boldsymbol{x} - \boldsymbol{\mu}) \cdot (\boldsymbol{\sigma}^{2} + \epsilon)^{-1/2} } \\
&= \frac{\partial (\boldsymbol{x} - \boldsymbol{\mu}) } {\partial \boldsymbol{x}_i} \cdot (\boldsymbol{\sigma}^{2} + \epsilon)^{-1/2} + (\boldsymbol{x}_i - \boldsymbol{\mu}) \cdot \frac{-1}{2} (\boldsymbol{\sigma}^{2} + \epsilon)^{-3/2} \frac{\partial \boldsymbol{\sigma}^2} {\partial \boldsymbol{x}_i}
\end{align}
$$

那么现在只需要分别求得 $\frac{\partial (\boldsymbol{x} - \boldsymbol{\mu}) } {\partial \boldsymbol{x}_i}$ 和 $\frac{\partial \boldsymbol{\sigma}^2} {\partial \boldsymbol{x}_i}$ 即可。

已知，$\boldsymbol{x} \in \mathbb{R}^{m \times d}, \boldsymbol{\mu} \in \mathbb{R}^{1 \times d}, \boldsymbol{\sigma}^2 \in \mathbb{R}^{1 \times d}$ ，有：

$$
\begin{align}
\mu_l &= \frac{1}{N} \sum_{k=1}^m {x_{kl}} \\
\sigma_l^2 &= \frac{1}{N} \sum_{k=1}^m (x_{kl} - \mu_l)^2 \\
\hat{x}_{kl} &= (x_{kl} - \mu_l) \, (\sigma_l^2 + \epsilon)^{-1/2}
\end{align}
$$

对于一个元素的偏导数如下：

$$
\begin{align}
\frac{\partial {(x_{kl} - \mu_l)}} {\partial x_{ij}} &= \delta_{ki} \delta_{lj} - \frac{1}{N} \delta_{lj} \\
\\
\frac{\partial \sigma_l^2} {\partial x_{ij}} &= \frac{1}{N} \sum_{k=1}^m {2(x_{kl} - \mu_{l}) (\delta_{ik} \delta_{lj} - \frac{1}{N} \delta_{lj})} \\
&= \frac{2}{N} (x_{il} - \mu_l)\delta_{lj} - \frac{2}{N^2} \sum_{k=1}^m { \delta_{lj} (x_{kl}-\mu_l)} \\
&= \frac{2}{N} (x_{il} - \mu_l)\delta_{lj} - \frac{2}{N} \delta_{lj} \left( \frac{1}{N}\sum_{k=1}^N {x_{kl} - \mu_l} \right) \\
&= \frac{2}{N} (x_{il} - \mu_l)\delta_{lj}
\end{align}
$$

将上述式子代入 $\frac{\partial \boldsymbol{\hat{x}}}{\partial \boldsymbol{x}_i}$ 中， 可得：

$$
\frac{\partial \hat{x}_{kl}} {\partial x_{ij}} = (\delta_{ki} \delta_{lj} - \frac{1}{N} \delta_{lj} ) (\sigma^2 + \epsilon)^{-1/2} - (x_{kl} - \mu_l) \cdot (\sigma^2 + \epsilon)^{-3/2} \cdot \frac{1}{N} (x_{il} - \mu_l)\delta_{lj}
$$

### Layer Normalization
