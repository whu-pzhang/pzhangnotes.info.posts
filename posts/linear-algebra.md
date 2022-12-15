---
title: 线性代数回顾
date: 2018-01-08
lastMod: 2018-04-07
author: pzhang
draft: true
markup: mmark
mathjax: true
categories:
  - 基础理论
tags:
  - 线性代数
slug: linear-algebra
---

线性代数知识点主要有以下这些：

>
- 矩阵和向量的基本运算
- 特殊矩阵
- 转置
- 对称矩阵
- 迹和范数
- 线性无关和秩
- 逆矩阵
- 正交矩阵（投影）
- 基本子空间
- 行列式
- 二次型和正定性
- 特征值和特征向量
- 矩阵微积分

<!--more-->

这里只会对我认为重要的部分详细记录。

**符号约定：**

- 标量(scalar)：常规小写Roman字符表示。定义一个标量时，通常会明确是哪种类型的数，$x \in \mathbb{R}$ 表示实数；$n \in \mathbb{N}$ 表示自然数。
- 向量(vector)：粗体小写Roman字符。$\boldsymbol{x} \in \mathbb{R}^n$ 表示向量 $\boldsymbol{x}$ 属于 $n$ 维向量空间。
- 矩阵(matrix)：大写斜粗体字符。$\boldsymbol{A} \in \mathbb{R}^{m \times n}$ 表示 $\boldsymbol{A}$ 为 $m$ 行 $n$ 列的实数矩阵。
- 张量(tensor)：表示超过两维的数组，一般用大写正粗体表示。$\mathbf{A}$


## 矩阵乘法

### 向量和向量乘法

- 内积

也称为点积。给定两个向量，$\boldsymbol{x, y} \in \mathbb{R}^n$，那么有：

$$
\boldsymbol{x^T y} =
\begin{bmatrix}
x_1 & x_2 &\cdots & x_n
\end{bmatrix}
\begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{bmatrix}
= \sum_{i=1}^n x_i y_i
$$

即 $\boldsymbol{x^T y} = \boldsymbol{y^T x}$

- 外积

给定两个向量，$\boldsymbol{x} \in \mathbb{R}^m,  \boldsymbol{y} \in \mathbb{R}^n$ ,
有 $\boldsymbol{x y^T} \in \mathbb{R}^{m \times n}$，且 $(xy^T)\_{ij} = x_i y_j$。例：

$$
\boldsymbol{x y^T} =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
\begin{bmatrix}
y_1 & y_2 & \cdots & y_n
\end{bmatrix} =
\begin{bmatrix}
x_1y_1 & x_1y_2 & \cdots & x_1y_n\\
x_2y_1  & x_2y_2 & \cdots & x_2y_n\\
\vdots & \vdots & \ddots & \vdots \\
x_m y_1 & x_m y_2 & \cdots & x_m y_n
\end{bmatrix}
$$


### 矩阵和向量乘法



### 矩阵和矩阵乘法

进行矩阵乘法的两矩阵形状必须匹配，矩阵 $\boldsymbol{A}$ 的列数必须和矩阵 $\boldsymbol{B}$ 的行数相等，若 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$, $\boldsymbol{B} \in \mathbb{R}^{n \times p}$，则 $\boldsymbol{C} \in \mathbb{R}^{m \times p}$，有：

$$
\boldsymbol{C = AB} \\
\Downarrow \\
C_{ij} = \sum A_{ik} B_{kj}
$$

当然，矩阵乘法除了以上计算公式，还有其他的理解办法，例如：行或者列的线性组合。

此外，还有一种矩阵“乘法”是两个矩阵对应元素的乘积，称为 **元素对应乘积**（element-wise product）或 **Hadamard 乘积**（Hadamard product），记为 $\boldsymbol{A \odot B}$。

**矩阵乘法性质：**

- 分配律：
  $$
  \boldsymbol{A(B+C) = AB + AC}
  $$

- 结合律：
  $$
  \boldsymbol{A(BC) = (AB)C}
  $$

- 不同于标量乘积和向量乘积，矩阵乘法不满足交换律。


## 基本概念

### 向量空间和子空间

将二维直角坐标系想象成一个2D空间，空间由无数个点组成，每个点是一个从原点出发的向量。在所有的向量中，有一组向量是特别的：$\hat i = (1,0), \hat{j} = (0,1)$ ，有了$\hat{i}$ 和 $\hat{j}$ ，空间上所有的向量都可以表示为它们的线性组合。将所有可以表示为给定向量线性组合的向量的集合，称为给定向量张成（span）的空间。

向量空间必须对对加法和乘法运算封闭，设$\boldsymbol{u}$ 和 $\boldsymbol{v}$ 为向量空间中的两个向量，$a,b,c$ 为任意实数，那么有：

1. $a\boldsymbol{u} + b\boldsymbol{v}$ 必须落在该空间中
2. $c \boldsymbol{u}$ 必须落在该空间中

被包含于上述向量空间中的空间称为子空间（子空间必定包含原点！）。

### 基本子空间

考虑实数线性系统 $\boldsymbol{Ax = b}$，其中 $\boldsymbol{A} \in \mathbb{R}^{m \times n}, \boldsymbol{x} \in \mathbb{R}^{n \times 1}, \boldsymbol{b} \in \mathbb{R}^{m \times 1}$。

$\boldsymbol{A}$ 的四个基本子空间：

- 矩阵 $\boldsymbol{A}$ 的列张成(span)的子空间称为 $\boldsymbol{A}$ 的列空间。$C(\boldsymbol{A}) \in \mathbb{R}^{m}$
- 矩阵 $\boldsymbol{A}$ 的行张成(span)的子空间称为 $\boldsymbol{A}$ 的行空间。$R(\boldsymbol{A}) \in \mathbb{R}^{n}$
- $\boldsymbol{Ax = 0}$ 的解张成的子空间称为  $\boldsymbol{A}$ 的零空间，记为 $N(\boldsymbol{A})$
- $\boldsymbol{A^Tx = 0}$ 的解张成的子空间称为  $\boldsymbol{A}$ 的左零空间，记为 $N(\boldsymbol{A^T})$

易知四个基本子空间的关系如下：

- $C(\boldsymbol{A}) \perp N(\boldsymbol{A^T})$

- $R(\boldsymbol{A}) \perp N(\boldsymbol{A})$

- $dimC(\boldsymbol{A}) + dimN(\boldsymbol{A}) = n $


### 线性无关

若一组向量中的任意一个都不能表示为其他向量的线性组合，则称这组向量线性无关。
$$
\boldsymbol{x_n} = \sum_i^{n-1} \alpha_i \boldsymbol{x_i}
$$
若存在不为零的实数 $\alpha_i$ 使得上式成立，则称向量组 ${\boldsymbol{x_1, x_2, \cdots, x_n}}$ 线性相关。

### 秩和维数

矩阵 $\boldsymbol{A}$ 的列秩（column rank）为其最大线性无关列向量的个数。

- $$


确定 $\boldsymbol{Ax = b}$ 是否有解相当于确定向量 $\boldsymbol{b}$ 是否在 $C(\boldsymbol{A})$ 中。为了使得 $\boldsymbol{Ax = b}$ 对任意向量 $\boldsymbol{b} \in \mathbb{R}^{m}$ 都有解，需要 $C(\boldsymbol{A})$ 构成整个 $\mathbb{R}^{m}$ 空间，这就需要 $\boldsymbol{A}$ 至少有 $m$ 列。



## 范数和迹

范数（norm）用来衡量向量或矩阵的大小。

### 向量范数

向量的范数 $L^p$ 定义如下：
$$
\Vert \boldsymbol{x} \Vert_p = \left( \sum_i |x_i|^p \right)^{\frac{1}{p}}
$$
范数是将向量映射到非负值的函数，衡量从原点到点 $\boldsymbol{x}$ 的距离，严格来讲，范数满足下列性质的任意函数：

- $f(\boldsymbol{x}) = 0 \rightarrow \boldsymbol{x = 0 }$

- $f(\boldsymbol{x+y}) \le f(\boldsymbol{x}) + f(\boldsymbol{y})$    

- $\forall \alpha \in \mathbb{R}, f(\alpha \boldsymbol{x}) = |\alpha| f(\boldsymbol{x})$

常见范数：
- $p=2$: 欧几里德范数(Euclidean norm)，经常简写为$\Vert \boldsymbol{x} \Vert$
$$
\| \boldsymbol{x} \|_2 = \sqrt {\sum_i |x_i|^2} = \sqrt{\boldsymbol{x^T x}}
$$

- $p=1$: $L^1$ 范数
$$
\| \boldsymbol{x} \|_1 = \sum_i |x_i|
$$

- $p=0$: $L^0$ 范数，表示非零元素的个数

- $p=\infty$: 最大范数$L^{\infty}$，表示向量中具有最大幅值的元素的绝对值：
$$
\| \boldsymbol{x} \|_{\infty} = \max_i |x_i|
$$

### 矩阵范数

最常用来衡量矩阵大小的范数为 **Frobenius 范数**（Frobenius norm）：

$$
\| \boldsymbol{A} \|_F  = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{tr(\boldsymbol{A^T A})}
$$

类似于向量的2范数。

### 迹

方块矩阵主对角元素的和称为迹。

$$
tr(\boldsymbol{A}) = \sum_i A_{ii}
$$

很多矩阵运算难以描述，而使用迹运算和矩阵乘法可以清晰的表示，例如前面的矩阵Frobenius范数。

迹运算有很多有用的性质：

- For $\boldsymbol{A} \in \mathbb{R}^{n \times n}, \quad tr(\boldsymbol{A^T}) = tr(\boldsymbol{A})$

- For $\boldsymbol{A, B} \in \mathbb{R}^{n \times n}, \quad  tr(\boldsymbol{AB}) = tr(\boldsymbol{A}) + tr(\boldsymbol{B})$

- For $\boldsymbol{AB}$ is square, $tr(\boldsymbol{AB}) = tr(\boldsymbol{BA})$

- For $\boldsymbol{A, B, C}$ such that $\boldsymbol{ABC}$ is square, $tr(\boldsymbol{ABC}) = tr(\boldsymbol{CAB}) = tr(\boldsymbol{BCA})$

## 矩阵分解

### 特征值和特征向量
## 矩阵求导

矩阵微积分([Matrix calculus](https://en.wikipedia.org/wiki/Matrix_calculus))是多元微积分的一种特殊表示形式，主要是为了方便计算以及更简洁的公式表示。矩阵微积分有两种布局：分子布局(Numberator-layout)和分母(Denominator-layout)，通常以向量对标量的导数写为列向量还是行向量予以区分。现有列向量如下：
$$
\boldsymbol{y} =
\begin{bmatrix}
y_1 & y_2 & \cdots & y_n
\end{bmatrix}^T
$$

- 分子布局

$$
\frac{\partial \boldsymbol{y}} {\partial x} =
\begin{bmatrix}
\frac{\partial y_1} {\partial x} & \frac{\partial y_2} {\partial x} & \cdots & \frac{\partial y_n} {\partial x}
\end{bmatrix}^T
$$

- 分母布局

$$
\frac{\partial \boldsymbol{y}} {\partial x} =
\begin{bmatrix}
\frac{\partial y_1} {\partial x} & \frac{\partial y_2} {\partial x} & \cdots & \frac{\partial y_n} {\partial x}
\end{bmatrix}
$$

我这里不做区分，求导结果与原矩阵同型，称为 **Mixed layout**。

$$
\begin{align}
\left( \frac{\partial \boldsymbol{a}}{\partial x} \right)_i &= \frac{\partial a_i}{\partial x} \\
\left( \frac{\partial a}{\partial \boldsymbol{x}} \right)_i &= \frac{\partial a}{\partial x_i}
\end{align}
$$

矩阵的导数：

$$
\left( \frac{\partial \boldsymbol{A}}{\partial x} \right)_{ij} = \frac{\partial A_{ij}}{\partial x}
$$

向量对向量的导数为：

$$
\left( \frac{\partial \boldsymbol{a}}{\partial \boldsymbol{b}} \right)_{ij} = \frac{\partial a_i}{\partial b_j}
$$

假定函数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ ，则有：

$$
\left ( \nabla_{\boldsymbol{x}} f(\boldsymbol{x}) \right)_i = \frac{ \partial f(\boldsymbol{x})} {\partial x_i}
$$

$f(\boldsymbol{x})$ 关于 $\boldsymbol{x}$ 的二阶导数是称为海森矩阵(Hessian matrix)的一个对称方阵：

$$
\left (\nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x}) \right)_{ij} = \frac{\partial^2 f(\boldsymbol{x})} {\partial x_i \partial x_j}, \quad \nabla_{\boldsymbol{x}}^2 f(\boldsymbol{x}) \in \mathbb{R}^{n\times n}
$$

向量和矩阵的导数满足乘法规则(product rule):

$$
\begin{align*}
\frac{\partial}{\partial \boldsymbol{x}} (\boldsymbol{x}^T \boldsymbol{a})
&=
\frac{\partial}{\partial \boldsymbol{x}} (\boldsymbol{a}^T \boldsymbol{x})
= \boldsymbol{a} \\
\frac{\partial}{\partial \boldsymbol{x}} (\boldsymbol{AB})
&=
\frac{\partial \boldsymbol{A}} {\partial \boldsymbol{x}} \boldsymbol{B} +
 \boldsymbol{A} \frac{\partial \boldsymbol{B}} {\partial \boldsymbol{x}}
\end{align*}
$$

由 $\mathbf{A^{-1}A = I}$ 及上式，有：

$$
\frac{\partial } {\partial x} (\boldsymbol{A}^{-1}) = - \boldsymbol{A}^{-1} \frac{\partial \boldsymbol{A}} {\partial x} \boldsymbol{A}^{-1}
$$

若 $x$ 为矩阵 $\boldsymbol{A}$ 的元素，则有：

$$
\frac{\partial}{\partial A_{ij}} \text{tr} (\boldsymbol{AB}) = B_{ji}\\
\frac{\partial}{\partial \boldsymbol{A}} \text{tr} (\boldsymbol{AB}) = \boldsymbol{B}^T
$$

进而有：

$$
\begin{align*}
\frac{\partial}{\partial \boldsymbol{A}} \text{tr} (\boldsymbol{A^TB}) &= \boldsymbol{B} \\
\frac{\partial}{\partial \boldsymbol{A}} \text{tr} (\boldsymbol{A}) &= \boldsymbol{I} \\
\frac{\partial}{\partial \boldsymbol{A}} \text{tr} (\boldsymbol{ABA^T}) &= \boldsymbol{A(B+B^T)}
\end{align*}
$$

矩阵范数的求导:

$$
\frac{\partial \| \mathbf{A} \|_F^2} {\partial \mathbf{A}} = \frac{\partial \text{tr} (\mathbf{AA}^T)} {\partial \mathbf{A}} = 2\mathbf{A}
$$

链式法则(chain rule)是计算复杂导数时的重要工具。若函数 $f$ 是 $g$ 和 $h$ 的复合，即 $f(x) = g(h(x))$ ,
则有：

$$
\frac{\partial f(x)} {\partial x} = \frac{\partial {g(h(x))}} {\partial h(x)} \cdot \frac{\partial h(x)} {\partial x}
$$

例如在计算 $ \| \mathbf{Ax-b} \|_2^2)$ 的梯度时，将 $\mathbf{Ax-b}$ 看作一个整体有：

$$
\begin{align*}
\nabla_{\mathbf{x}} J(\mathbf{x}) &= \frac{\partial }{\partial \mathbf{x}} (\mathbf{Ax-b})^T(\mathbf{Ax-b}) \\
&= \frac{\partial (\mathbf{Ax-b})} {\partial \mathbf{x}} \cdot 2(\mathbf{Ax-b}) \\
&= 2 \mathbf{A}^T \mathbf{(Ax-b)}
\end{align*}
$$


**常见技巧及注意事项**

- 对下角标形式和矩阵表示形式的转化要敏感，常见的有：

  -  $\sum u_i v_i = \mathbf{u}^T\mathbf{v} = \mathbf{v}^T\mathbf{u} = \text{tr} (\mathbf{u}^T\mathbf{v})$
  -  $\sum_{i,j} a_{ij} b_{ij} = \text{tr} (\mathbf{AB}^T)$

- 将实数看作是 $1*1$ 矩阵的迹来给式子套上 $\text{tr}$ ，然后利用迹运算的性质

- 若函数表达式中，某个变量出现了多次，可以每次单独计算其中一个的导数，然后再将结果相加

  计算 $\nabla (\mathbf{x}^T \mathbf{Ax}) $ 时，可先将 $\mathbf{x}$ 当作不同的变量，有
  $$
  \nabla_{\mathbf{x}} (\mathbf{x}^T \mathbf{Ax}) = \mathbf{Ax} + \mathbf{A}^T \mathbf{x} = (\mathbf{A+A}^T) \mathbf{x}
  $$
  ​
