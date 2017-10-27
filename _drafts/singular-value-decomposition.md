---
title: singular-value-decomposition
author: pzhang
Date:
categories: math
tags:
---

奇异值分解



## 基本概念

以 $m \times n$ 矩阵 $\mathbf{A}$ 为说明对象

**列空间和行空间**

矩阵 $\mathbf{A}$ 的列向量生成的空间称为 $\mathbf{A}$ 的列空间，用 $Col \, \mathbf{A}$ 表示。由于每个列向量都含有 $m$ 个元素，$Col \, \mathbf{A}$ 为 $\mathbf{R}^m$ 的子空间。

同样的，行空间为矩阵 $\mathbf{A}$ 的行向量生成的空间，记为 $Row \, \mathbf{A}$ 。其为 $\mathbf{R}^n$ 的子空间。

**零空间**





## 正交矩阵

满足 $\mathbf{A^T A = I}$  的方块矩阵称之为正交矩阵。从定义出发，设 $\mathbf{A}_i$ 为组成 $n$ 阶方阵的列向量，那么有：
$$
\mathbf{A^T A} = 
\begin{bmatrix}
A_1^T \\
A_2^T \\
\vdots \\
A_n^T
\end{bmatrix}

\begin{bmatrix}
A_1 & A_2 &  \cdots & A_n
\end{bmatrix}

= 
\begin{bmatrix}
A_1^T A_1 & A_1^T A_2 & \cdots & A_1^T A_n \\
A_2^T A_1 & A_2^T A_2 & \cdots & A_2^T A_n \\
\vdots & \vdots & \ddots & \vdots \\
A_n^T A_1 & A_n^T A_2 & \cdots & A_n^T A_n
\end{bmatrix}
= \mathbf{I}
$$
即
$$
A_i^T A_j =
\begin{cases}
1, & i=j \\
0, & i \neq j
\end{cases}
$$
也就是说方阵 $\mathbf{A}$ 的列向量为互相正交的单位向量。