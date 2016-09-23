---
title: 特征值和奇异值
author: pzhang
date: 2016-09-21 20:54:31
categories: 地震学基础
tags: [MATLAB]
---

方阵$A$的特征值(eigenvalue)及特征向量(eigenvector)，分别为满足下式的标量$\lambda$及
非零向量$x$

$$Ax = \lambda x$$

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
