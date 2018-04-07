---
title: 线性代数基础知识
author: pzhang
date: 2017-10-03
draft: true
category: 理论基础
tags:
  - 线性代数
---

奇异值分解



## 基本概念

以 $m \times n$ 矩阵 $\mathbf{A}$ 为说明对象

**列空间和行空间**

矩阵 $\mathbf{A}$ 的列向量生成的空间称为 $\mathbf{A}$ 的列空间，用 $Col \, \mathbf{A}$ 表示。由于每个列向量都含有 $m$ 个元素，$Col \, \mathbf{A}$ 为 $\mathbf{R}^m$ 的子空间。

同样的，行空间为矩阵 $\mathbf{A}$ 的行向量生成的空间，记为 $Row \, \mathbf{A}$ 。其为 $\mathbb{R}^n$ 的子空间。

**零空间**

齐次方程 $\mathbf{A}\boldsymbol{x}=\mathbf{0}$ 的所有解生成的空间称为 $\mathbf{A}$ 的零空间，记为 $Nul \, \mathbf{A}$ 。

**基**

$\mathbb{R}^n$ 中的一组线性无关向量生成子空间 $H$ ，那么这组向量即为子空间 $H$ 的一组基。

**维数**

非零子空间 $H$ 的任意一组基中的向量个数称为维数，记为 $dim \,H $ 。零子空间的维数为零。

$\mathbb{R}^n$ 空间的每个基由 $n$ 个向量，其维数为 $n$ 。$\mathbb{R}^3$ 中一个经过 $\mathbf{0}$ 的平面是2维的，一条经过 $\mathbf{0}$ 的直线是1维的。

**秩**

矩阵 $\mathbf{A}$ 的列空间的维数称为秩，记作 $rank \, \mathbf{A}$ 。也即是 $\mathbf{A}$ 主元的数目。



## 正交矩阵

满足 $\mathbf{A^T A = I}$  的方块矩阵称之为正交矩阵。从定义出发，设 $\mathbf{A}_i$ 为组成 $n$ 阶方阵的列向量，那么有：

$$
A_i^T A_j =
\begin{cases}
1, & i=j \\
0, & i \neq j
\end{cases}
$$
也就是说方阵 $\mathbf{A}$ 的列向量为互相正交的单位向量。


$$
对称矩阵 \iff 二次型矩阵 \Longleftrightarrow 二次型
$$
