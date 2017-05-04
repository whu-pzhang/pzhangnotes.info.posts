---
title: NumPy学习笔记
author: pzhang
date: 2017-05-03
tags: [numpy]
categories: Programming
---

[NumPy](http://www.numpy.org/) 为 Python下科学计算的基础包。

# 知识预备

显而易见，学习NumPy前，你得了解一些 Python 的基础知识。如果你想复习一下，请看
[Python Tutorial](https://docs.python.org/3/tutorial/)

学习之前，需要安装 Python 和 NumPy, 这里推荐 [Anaconda](https://www.continuum.io/downloads)。
详细安装过程，可以参照[Python科学计算环境Anaconda](/python-anaconda.html)。

<!--more-->

# 基础篇

NumPy 的操作对象主要为同种对象的多维数组。在 NumPy 中维度(dimensions) 叫做
轴(axes)，轴的个数称为秩(rank)。

例如，在3D空间一个点的坐标`[1, 2, 3]` 为一个秩为1的数组，因为它只有一个轴，
轴的长度为3。又例如，在以下例子中，数组的秩为2（它有两个维度），第一维长为
2，第二维长度为3.
```
[[1., 0., 0.],
 [0., 1., 2.]]
```



# 参考
1. [NumPy Tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
