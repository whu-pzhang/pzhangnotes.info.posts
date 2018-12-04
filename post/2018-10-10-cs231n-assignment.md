---
title: cs231n作业笔记
author: pzhang
date: 2018-10-10
lastMod: 2018-10-10
markup: mmark
mathjax: true
categories:
  - 计算机视觉
tags:
  - Python
  - NumPy

draft: true
slug: kalman-filter
---

## 课程简介

这段时间刷了一遍斯坦福的[cs231n](http://cs231n.stanford.edu/)计算机视觉课程。其作业难度相比Andrew Ng在Coursera上的机器学习而言大得多。

Assignment主页： [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)

今年Spring 2018的作业分为3个部分，都是在Python3环境下进行的。

[Assignment #1: Image Classification, kNN, SVM, Softmax, Neural Network](https://cs231n.github.io/assignments2018/assignment1/)

[Assignment #2: Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets](https://cs231n.github.io/assignments2018/assignment2/)

[Assignment #3: Image Captioning with Vanilla RNNs, Image Captioning with LSTMs, Network Visualization, Style Transfer, Generative Adversarial Networks](https://cs231n.github.io/assignments2018/assignment3/)


Assignment1从最简单的kNN分类器开始，到基本的线性分类器，逐步实现SVM以及Softmax分类器，直到最后需要手写一个简单的双层神经网络分类器。


我的代码实现[whu-pzhang/cs231n](https://github.com/whu-pzhang/cs231n).

<!--more-->

## assignmet1

作业1的前导课程主要包括课程介绍、对数据驱动方法的介绍，kNN方法，线性分类方法（SVM以及Softmax）以及神经网络方法（多层感知机和反向传播算法）。
因此，作业1的内容也是基于这几个方面的。

首先需要说明的是，所有分类方法的使用均分为两步：`train()` 和 `predict()`

### kNN

kNN（k-Nearest Neighbor）方法

- 训练阶段： 简单的记住训练集数据
- 预测阶段： 测试数据分别和训练集中的所有数据计算相似度（以距离为度量），`k=1`时，取最相似的训练数据的标签作为该测试数据的标签；若`k>1`，则选择最相似的`k`个训练数据标签中出现次数最多的类别（多数投票法，vote）作为预测输出。

kNN 方法的主要计算量在计算测试数据与训练集数据的距离这里，因此距离计算的高效性十分重要。这部分的作业也是循序渐进，要求分别实现：双循环计算距离，单循环计算距离以及完全向量化的无循环计算距离。

1. 双循环：每个测试数据和每个训练数据分别计算，可以使用`np.linalg.norm()`来计算距离。
2. 单循环：每个测试数据和整个训练集分别计算
3. 无循环：令测试集为$P$，其维度为 $M \times D$，训练集$C$的维度为 $N \times D$，其中 $M$ 为测试集样本数， $D$为每个样本的维度， $N$为训练集样本数。记 $P_i$为其中一个测试样本，$C_j$ 为一个任意一个训练样本

$$
P_i = [P_{i1}, P_{i2}, \cdots, P_{iD}] \\
C_j = [C_{j1}, C_{j2}, \cdots, C_{jD}]
$$

那么 $P_i$ 和 $C_j$ 的距离的平方为：

$$
\begin{align}
d(P_i, C_j) &= \sum_{k=1}^D {(P_{ik} - C_{jk})^2} \\
&= \Vert P_i \Vert^2 + \Vert C_j \Vert^2 - 2 P_i C_j^T
\end{align}
$$

上述结果是得到是最终的距离矩阵（$M \times N$）其中的一个元素。那么我们可以推广得到距离矩阵的其中一行为：

$$
\begin{align}
\boldsymbol{R}_i &= [\Vert P_i \Vert^2 \quad \Vert P_i \Vert^2 \quad \cdots \quad \Vert P_i \Vert^2 ] + [\Vert C_1 \Vert^2 \quad \Vert C_2 \Vert^2 \quad \cdots \quad \Vert C_N \Vert^2] - 2 P_i [C_1^T \quad C_2^T \quad \cdots \quad C_N^T] \\
&= [\Vert P_i \Vert^2 \quad \Vert P_i \Vert^2 \quad \cdots \quad \Vert P_i \Vert^2 ] + [\Vert C_1 \Vert^2 \quad \Vert C_2 \Vert^2 \quad \cdots \quad \Vert C_N \Vert^2] - 2 P_i C^T]
\end{align}
$$


继而，可得距离矩阵为：

$$
\begin{align}
\boldsymbol{R} &= \begin{bmatrix}
\Vert P_1 \Vert^2 & \Vert P_1 \Vert^2 & \cdots & \Vert P_1 \Vert^2 \\
\Vert P_2 \Vert^2 & \Vert P_2 \Vert^2 & \cdots & \Vert P_2 \Vert^2 \\
\vdots & \vdots & \ddots & \vdots \\
\Vert P_M \Vert^2 & \Vert P_M \Vert^2 & \cdots & \Vert P_M \Vert^2
\end{bmatrix} +
\begin{bmatrix}
\Vert C_1 \Vert^2 & \Vert C_2 \Vert^2 & \cdots & \Vert C_N \Vert^2 \\
\Vert C_1 \Vert^2 & \Vert C_2 \Vert^2 & \cdots & \Vert C_N \Vert^2 \\
\vdots & \vdots & \ddots & \vdots \\
\Vert C_1 \Vert^2 & \Vert C_2 \Vert^2 & \cdots & \Vert C_N \Vert^2
\end{bmatrix} - 2 \boldsymbol{P} \boldsymbol{C}^T
\end{align}
$$

注意，这里计算距离的时候，没有计算其平方跟，这是因为进行距离比较时，计不计算平方跟不影响结果，平方根运算属于冗余计算。

最终计算的时候利用广播即可。代码如下：

```python
dists += np.sum(X**2, axis=1).reshape(num_test, 1)
dists += np.sum(self.X_train**2, axis=1).reshape(1, num_train)
dists -= 2 * (X @ self.X_train.T)
```

计算得到距离后，然后就是预测了。可利用 `np.argsort()` 获取距离最近的前`k`个训练样本，然后采用 `np.bincount()` 与 `np.argmax()` 结合，得到高票标签作为预测输出。

最后部分是利用交叉验证得到kNN中的最佳的`k`值。

### SVM

第二部分是多分类的SVM算法。

线性分类器表示如下：

$$
f(x_i; \boldsymbol{W}, b) = \boldsymbol{W} x_i + b
$$

$f(x_i; \boldsymbol{W}, b)$ 称为单个样本的分数（score），首先需要计算线性SVM的合页损失函数：

$$
L = \frac{1}{N} \sum_{i=1}^N {L_i} + \lambda R(\boldsymbol{W})
$$

其中，

$$
L_i = \sum_{j \ne y_i} max(0, s_j - s_{y_i} + \Delta)
$$

式中，$s_{y_i}$ 表示正确类别的分数，$s_j$则表示其他类别对应的分数。也就是说只有错误的分类才会产生loss，正确分类的loss为0。

这里的 $\Delta$ 表示最大间隔，需要注意的是，它的值不需要通过交叉验证来确定，大多数情况下，令 $\Delta = 1.0$ 即可。
这是因为 $\Delta$ 和 $\lambda$ 一样控制着数据损失和正则化损失之间的权衡，而权重矩阵 $\boldsymbol{W}$ 的缩放可以直接影响各个类别分数之间的差异，因此分数之间边界的精确值（$\Delta=1$ 或 $\Delta=100$），在某种程度上是无意义的。

主要代码如下：

```python
margins = np.maximum(0, scores - corrent_scores + 1) # delta = 1
margins[np.arange(num_train), y] = 0.0
```

接下来是权重矩阵$\boldsymbol{W}$梯度的计算。

$$
\begin{cases}
\nabla_{w_{y_i}} L_i = - \left ( \sum_{j \ne y_i} {\mathbb{I} (margin_j > 0)} \right) x_i \\
\nabla_{w_{j}} L_i = \mathbb{I} (margin_j > 0) x_i
\end{cases}
$$

其中 $\mathbb{I}$ 为指示函数，条件为真时，其值为1；反之为零。上式表达的意思就是：每个 $margin > 0$  的分类会对正确类别产生 $-x_i$ 的贡献，而对于错误分类产生一个 $x_i$ 的贡献。对某个样本，所有 $margin>0$ 的错误分类（正确分类 $margin$ 为 0）对正确分类共产生 $NUM(margin > 0)​$ 次贡献，而对错误分类则只产生一次贡献。

根据反向传播原理（链式法则）：

$$
\nabla_{\boldsymbol{W}} L = \frac{\partial L} {\partial \mathbf{s}} \frac{\partial \mathbf{s}} {\partial \boldsymbol{W}}
$$

其中，$\frac{\partial \mathbf{s}} {\partial \boldsymbol{W}} = \boldsymbol{X}^T$，因此只需要构造 $\frac{\partial L} {\partial \mathbf{s}}$ 即可。

```python
dS = np.zeros((num_train, num_classes))
dS[margins > 0] = 1  # only margin > 0 contribute to gradient
dS[np.arange(num_train), y] -= np.sum(coeff_mat, axis=1)

dW = X.T @ dS
```

另外，计算损失和梯度时不要漏掉正则项即可。

损失函数和梯度计算搞定后，小批量SGD就很简单了，直接根据学习率更新参数即可。

### Softmax

这部分是基于线性模型的softmax损失分类器，与SVM不同的是，损失函数里的子项是交叉熵（cross-entropy）：

$$
L_i = -\log \left( \frac{e^{f_{y_i}}} {\sum_{j} {e^{f_j}}} \right) =
-f_{y_i} + \log (\sum_{j} {e^{f_j}})
$$

这里计算loss若采用的是第一种表示形式，那么需要注意数值稳定性的问题：分子分母中都有指数项。实现时，可先将得到的值shift一下，这样不会改变最终的结果，但是数值稳定。

$$
\frac{e^{f_{y_i}}} {\sum_{j} {e^{f_j}}} = \frac{C e^{f_{y_i}}} {C \sum_{j} {e^{f_j}}} = \frac{e^{f_{y_i} + \log C}} {\sum_{j} {e^{f_j + \log C}}}
$$

上式中，取 $\log C = - \max_j f_j$ 的话，实现如下：

```python
scores = X.dot(W)
scores -= np.max(scores, axis=1, keepdims=True)  # N X 1
```

对权重矩阵的梯度如下：

$$
\nabla_\boldsymbol{W} L = \left( - \mathbb{I} (j = i) +  \frac{e^{\hat{f}_{y_i}}} {\sum_{j} {e^{\hat{f}_j}}} \right) x_i
$$

即样本对正确分类的贡献比错误分类的贡献小 $x_i$。具体实现和SVM类似：

```python
normalized_scores = np.exp(
    scores) / np.sum(np.exp(scores), axis=1, keepdims=True)  # N X C
normalized_scores[np.arange(num_train), y] += -1
dW = X.T @ normalized_scores
```

### 两层神经网络

这部分需要完成一个很简单的两层神经网络。包含一个隐藏层，激活函数为`ReLU`，使用Softmax分类损失函数。

loss 的计算和前面SVM以及Softmax类似，不再赘述。

主要是梯度的计算过程。应用反向传播原理，理清顺序即可，最好画个示意图。

```python
dscores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
dscores[np.arange(N), y] += -1
dscores /= N
grads['W2'] = h1_output.T @ dscores + 2.0 * reg * W2
grads['b2'] = np.sum(dscores, axis=0)

dh = dscores @ W2.T
dh_ReLU = (h1_output > 0) * dh
grads['W1'] = X.T @ dh_ReLU + 2.0 * reg * W1
grads['b1'] = np.sum(dh_ReLU, axis=0)
```

### 更高级别的表示

这部分主要是将 Histogram of Oriented(HoG) 特征和 color histogram 预先从图像中提取出来，然后作为特征加入到神经网络的输入中，从而达到提高预测准确率的目的。

练习部分主要是调参，尝试不同的学习率和正则化强度后，达到最好的预测准确率即可。


至此，作业1的全部内容就完成了！
