---
title: 概率论回顾
author: pzhang
date: 2018-02-08
lastMod: 2018-04-07
draft: true
markup: mmark
mathjax: true
categories:
  - 基础理论
tags:
  - math
  - 概率论
slug: probability
---

概率论基础回顾。

<!--more-->

## 概率公理

## 概率分布

随机变量是可以随机取不同值的变量。可以是离散的或者连续的。

概率分布（probability distribution）用来描述随机变量或一簇随机变量在每一个可能取到的状态的可能性大小。

### 离散型随机变量和概率质量函数

离散型随机变量的概率分布用**概率质量函数**(probability mass function, PMF)来描述。通常用 $P$ 来表示。PMF可以同时作用于多个随机变量，多个变量的概率分布称为**联合概率分布**(joint probability distribution)。 $P(x, y)$ 表示 $x$ 和 $y$ 同时发生的概率。

- $P$ 的定义域必须是 $\rm{x}$ 所有可能状态的集合。
- $\forall x \in {\rm x}, \quad  0 \le P(x) \le 1$
- $\sum_{x \in {\rm x}} P(x) = 1$

### 连续型随机变量和概率密度函数

**概率密度函数**(probability density function, PDF)用来描述连续性随机变量。一般用 $p$ 表示，满足如下性质：

- $p$ 的定义域必须是 $\mathrm{x}$ 所有可能状态的集合。
- $\forall x \in \rm{x}, p(x) \ge 0$ ，注意并不要求 $p(x) \le 1$。
- $\int p(x) {\rm d}x = 1$

$p(x)$ 没有直接给出特定状态的概率，其给出的是在面积为 $\delta x$ 无限小的区域内的概率为 $p(x) \delta x$ 。对PDF求积分获得区间内真实概率。

## 边缘概率

定义在子集上的概率分布称为**边缘概率分布**(marginal probability distribution)。**‘‘边缘概率’’** 的名称来源于手算边缘概率的计算过程。当 $P(x, y)$ 的每个值被写 在由每行表示不同的 $x$ 值，每列表示不同的 $y$ 值形成的网格中时，对网格中的每行 求和是很自然的事情，然后将求和的结果 $P(x)$ 写在每行右边的纸的边缘处。

假设有离散随机变量 $\mathrm{x}$ 和 $\mathrm{y}$ ，已知 $P(\mathrm{x, y})$ ，那么可以依据**求和规则**来计算 $P(\mathrm{x})$ :

$$
\forall x \in \mathrm{x}, \, P(\mathrm{x}=x) = \sum_{y} P(\mathrm{x}=x, \mathrm{y}=y)
$$

对于连续型变量，用积分代替求和即可：

$$
p(x) = \int p(x, y) \mathrm{d} y
$$

## 条件概率及贝叶斯定理

### 条件概率

在给定其他事件发生时我们感兴趣的某个事件出现的概率叫做**条件概率(conditional probability)**。将给定事件 $X$ 时，事件 $Y$ 发生的条件概率记为 $P( Y|X)$，条件概率和联合概率的关系表示如下：

$$
P(X, Y) = P(X | Y) P(Y) \\
P(Y, X) = P(Y|X) P(X)
$$

任何多维随机变量的联合概率分布都可以分解为只有一个变量的条件概率相乘的形式：

$$
P(\mathrm{x^{(1)}, \cdots, x^{(n)}}) = P(\mathrm{x}^{(1)}) \prod_{i=2}^n P(\mathrm{x^{(i)} | x^{(1)}, \cdots, x^{(i-1)}})
$$

此即概率论中的**乘法规则**。例如：

$$
\begin{align}
P(a, b, c) &= P(a| b, c) P(b, c) \\
&= P(a|b, c) P(b|c) P(c)
\end{align}
$$

### 贝叶斯定理



## 期望、方差和协方差

函数 $f(x)$ 关于分布 $(x)$ 的期望 (expectation) 为：

$$
E[f(x)] = \int f(x) p(x) \mathrm{d} x
$$
