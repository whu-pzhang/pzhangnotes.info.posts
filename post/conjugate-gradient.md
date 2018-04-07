---
title: 共轭梯度法
author: pzhang
date: 2017-07-10
draft: true
category: 理论基础
tags:
  - 算法
  - 基础

slug: conjugate-gradient
---

共轭梯度法笔记！

<!--more-->

## 符号约定

大写字母表示矩阵(matrix)，小写字母表示向量(vector)，希腊字符表示标量(scalar)。

## 数学定义

### 正定矩阵

对任意一个非零向量 $x$，矩阵 $A$ 若满足：
$$ x^T A x > 0$$
则称矩阵$A$为正定矩阵(positive matrix)。


### 二次型

对于二次型方程：

$$ f(x) = \frac{1}{2} x^T A x - b^T x + c$$

若 $A$ 为对称正定矩阵，那么线性方程 $Ax = b$ 的解正好使得 $f(x)$ 最小。

证明如下：

令 $\alpha = x^T A x$，那么有：
$$
\alpha = \sum_{i=1}^n \sum_{j=1}^n {A_{ij} \, x_i \, x_j}
$$
对 $\alpha$ 求偏导有：
$$
\begin{aligned}
\frac{\partial{\alpha}}{\partial{x_k}} &= \sum_{j=1}^n {A_{ij} \, \delta_{ik} \, x_j} + \sum_{i=1}^n {A_{ij} \, x_i \delta_{jk} } \\
                                       &= \sum_{j=1}^n {A_{kj} \, x_j} + \sum_{i=1}^n {A_{ik} \, x_i } \\
                                       &= x^T A^T + x^T A
\end{aligned}
$$

那么对$f(x)$ 求偏导有：
$$
\frac{\partial{f(x)}}{\partial{x}} = \frac{1}{2} x^T \left( A^T + A \right) - b^T
$$
若 $A$ 为对称矩阵，则有：
$$
\frac{\partial{f(x)}}{\partial{x}} = A x - b
$$

令梯度为零，即 $\frac{\partial{f(x)}}{\partial{x}} = 0$，就得到：
$$
Ax = b
$$

因此，该方程的解为 $f(x)$ 的临界点。若 $A$ 同时为正定矩阵，该解就使得 $f(x)$ 取
最小值。所以，我们可以通过找到二次型 $f(x)$ 的最小值 $x$ 来获得 $Ax=b$ 的解。

## 最速下降法

一些定义：

- 误差(error): $e_i = x_i - x$，表示离真实解的距离
- 残差(residual):  $r_i = b- A x_i$，表示当前值离正确 $b$ 值的距离

可以发现： $r_i = -A e_i$，也就是说矩阵 $A$ 将误差 $e$ 传递给了 $b$ 所在的空间。
更为重要的特性是：
$$r_i = b - A x_i = -f^{\prime}(x_i)$$

残差即为下降最快的方向！！这一点要牢记！

在最速下降法中，我们选取任意一点作为初始值，假设初始值为 $x_0$，那么

$$ x_{1} = x_{0} + \alpha \, r_{0} $$

那么怎么确定步长 $\alpha_i$ 和 方向 $p_{i}$ 呢？

我们选择 $f(x)$ 下降最快的方向，即其梯度 $f^{\prime}(x_i)$ 的反方向:
$$ r_i = -f^{\prime}(x_i) = b - A x_i$$
同时也是残差(residual)的方向。

## 特征向量

矩阵 $B$ 的特征向量 $v$

### Jacobi 迭代

$A = D + E$

## 地震勘探应用

线性正演算子：

$$ L \, m = d $$

其中 $L$ 为线性正演算子，$m$ 为模型向量，$d$ 为数据向量。

偏移算子为正演算子 $L$ 的伴随：
$$
\hat{m} = L^H \, d = L^H L \, m
$$

矩阵 $L^H L$ 为 Hessian 矩阵，$m$ 为真实模型，$\hat{m}$ 为偏移得到的模型。

可以看出，偏移结果只是真实地球模型的一个模糊(blur)表示。

为了消除假象(incomplete data or simulateneous-source data)，我们可以最小化以下
目标方程得到 $m$ ：
$$
f(m) = \left|| L m - d \right||_2^2 + \epsilon \left|| m \right||_2^2
$$


## 参考
- [Conjugate gradient method](https://en.wikipedia.org/wiki/Conjugate_gradient_method)
- [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](http://www.cs.cmu.edu/~./quake-papers/painless-conjugate-gradient.pdf)
