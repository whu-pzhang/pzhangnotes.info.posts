---
title: 线性代数基础
author: pzhang
date: 2016-11-30 09:54:31
categories: 地震学基础
tags: [MATLAB, Mada]
---

回顾线性代数的基础知识！

<!--more-->

## 向量空间

## 相似矩阵

同一个线性变换的不同基的描述矩阵，称为相似矩阵。

也就是说，一组相似矩阵都是同一个线性变换的描述。

## 特征值和奇异值

### 特征向量

特征值和特征向量从线性空间的角度更好理解。

N 维矩阵对应着一个 N 维线性空间到自身的线性变换。
对一个 **对称方阵** 进行特征值分解，就是产生了该线性空间
上的 N 组标准正交基，也就是 N 个特征向量，而特征值就对应着矩阵投影到这 N 组
正交基上的长度。

特征值越大，说明矩阵在对应的特征向量上的方差越大，功率越大，信息量越多。

特征向量的引入就是为了选取一组好的正交基。

从数学上描述，方阵 $A$ 的特征值(eigenvalue)和特征向量(eigenvector)，为满足下式
的非零标量 $\lambda$ 和非零向量 $x$

$$A x = \lambda x$$

### 奇异值

方阵或非方矩阵$A$的奇异值(singular value)及奇异值向量(singular vector)对，为分别满足以下
两式的非负标量 $\sigma$ 以及非零向量 $u$ 和 $v$ ：
$$
\begin{aligned}
Av & = \sigma u, \\
A^H u &= \sigma v
\end{aligned}
$$

这里，$A^H$的上标代表矩阵的埃尔米特转置(Hermitan transpose)，及复矩阵的共轭转置。若$A$
为实矩阵，那么$A^T$表示矩阵的转置。 MATLAB 中这都用 ```A'``` 表示。

需要说明的是，特征值的完整翻译应该为本征值(own value)，eigenvalue是德文
“eigenvert”的非完整翻译。

奇异矩阵

> 行列式等于零的方阵称为奇异矩阵。即 |A| != 0
> 而|A|!=0表示矩阵A可逆，我们可以知道： 可逆矩阵就是非奇异矩阵，非奇异矩阵也是可逆矩阵。

方阵的特征值-特征值向量方程可以写为：
$$(A - \lambda I) x = 0, x \ne 0$$
这意味着 $A-\lambda I$奇异，有
$$det(A-\lambda I) = 0$$
该方程即为矩阵$A$的特征方程(characteristic equation)或称为特征多项式(characteristic polynomial)。
特征多项式的次数(degree)即为矩阵的阶(order)。
