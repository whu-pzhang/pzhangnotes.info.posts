---
title: NumPy学习笔记
author: pzhang
date: 2017-05-03
tags: [numpy]
categories: Programming
---

[NumPy](http://www.numpy.org/) 为 Python下科学计算的基础包。

## 简介

NumPy 的出现主要是弥补 Python 数值计算的不足。为 Python 科学计算提供了强大的支撑。

NumPy 主要提供了两种对象：ndarray(N-dimensional array)和 ufunc(Unversal function)。

<!--more-->

## 基础篇

一个 ndarray (以下称为数组)是包含同种元素类型的多维数组容器。在 NumPy 中维度(dimensions) 叫做轴(axes)，轴的个数称为秩(rank)。

例如，在 3D 空间一个点的坐标`[1, 2, 3]` 为一个秩为1的数组，因为它只有一个轴，轴的长度为3。又例如，在以下例子中，数组的秩为2（它有两个维度），第一维长为2，第二维长度为3.

    [[1., 0., 0.],
     [0., 1., 2.]]

更多重要 ndarray 对象属性有：

| 属性                 | 描述                        |
| :----------------- | :------------------------ |
| `ndarray.flags`    | 数组内存布局信息                  |
| `ndarray.strides`  | 遍历数组时每个维度需要跳过的字节数组成的元组    |
| `ndarray.ndim`     | 数组轴的个数，即 秩                |
| `ndarray.shape`    | 数组的维度，元组的长度为秩，即维度或ndim属性  |
| `ndarray.size`     | 数组的总个数，等于shape属性中元组元素的乘积。 |
| `ndarray.dtype`    | 数组元素类型                    |
| `ndarray.itemsize` | 每个元素的字节大小                 |
| `ndarray.data`     | 包含实际数组元素的缓冲区              |

**一个例子**

```python
>>> import numpy as np
>>> x = np.array([[1, 2, 3], [4, 5, 6]])
>>> type(x)
<type 'numpy.ndarray'>
>>> x.dtype
dtype('int64')
>>> x.shape
(2, 3)
>>> x.flags
  C_CONTIGUOUS : True
  F_CONTIGUOUS : False
  OWNDATA : True
  WRITEABLE : True
  ALIGNED : True
  UPDATEIFCOPY : False
>>> x.strides
(24, 8)
```

后面代码都默认包含

``` python
import numpy as np
```



### 构建数组

创建数组主要有两种方式：利用`array`函数接受 Python 列表或元组来创建，这种情况是在数组内容已知的前提下；更多情况是，我们需要创建特定类型的数组，这时候需要用到其它的函数。

利用`array`手动创建：

``` python
>>> a = np.array([1, 2, 3])
>>> a
array([1, 2, 3])
>>> b = np.array([[1, 2, 3],[4, 5, 6]])
>>> b
array([[1, 2, 3],
       [4, 5, 6]])
```

此外，常见的创建数组的函数包括：

| 函数         | 功能                          |
| :--------- | --------------------------- |
| `zeros`    | 创建全为0的数组                    |
| `empty`    | 创建随机初始化的数组                  |
| `ones`     | 创建全为1的数组                    |
| `arange`   | 通过[start, stop, step]创建浮点数组 |
| `linspace` | 以[start, stop, num]创建浮点数组   |

由于浮点数有限精度的限制，通常利用` linspace` 创建浮点数组更好。

除了上述函数以外，还有`zeros_like`, `ones_like`,`empty_like` ,`fromfunction`和`fromfile`等函数也能来创建数组。详见[Array creation routines](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-creation.html#routines-array-creation).

###基本操作

数组的算术运算都是对数组元素逐个进行的(elmentwise)。这种算术运算称为 Universal function。

``` python
>>> a = np.array([10, 20, 30, 40])
>>> b = np.arange(4)
>>> b
array([0, 1, 2, 3])
>>> a - b
array([10, 19, 28, 37])
>>> b**2
array([0, 1, 4, 9])
>>> b * np.sin(a)
array([-0.        ,  0.91294525, -1.97606325,  2.23533948])
```

####索引，切片

索引和切片与 Python 中列表和元组的方式是一样的。不再赘述。

#### 维度操作

数组的`shape`属性给出了每个轴(axis)的元素个数。数组的 shape 可以通过函数来改变。

``` python
>>> a = np.floor(10*np.random.random((3, 4)))
>>> a
array([[ 5.,  6.,  2.,  5.],
       [ 9.,  3.,  3.,  4.],
       [ 7.,  2.,  2.,  8.]])
>>> a.shape
(3, 4)
>>> a.ravel() 	# 返回展平后的数组
array([ 5.,  6.,  2.,  5.,  9.,  3.,  3.,  4.,  7.,  2.,  2.,  8.])
>>> a.reshape(6,2) # 改变 shape
array([[ 5.,  6.],
       [ 2.,  5.],
       [ 9.,  3.],
       [ 3.,  4.],
       [ 7.,  2.],
       [ 2.,  8.]])
>>> a.T  # 转置
array([[ 5.,  9.,  7.],
       [ 6.,  3.,  2.],
       [ 2.,  3.,  2.],
       [ 5.,  4.,  8.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
```

 值得注意的是，上述三个命令都是返回一个视图，对数据的修改都会直接影响原数组。

## 拷贝和视图

当我们操作数组时，有时会拷贝到新的数组，有时不拷贝。具体有三种情况：

### 完全不拷贝

对于简单的赋值操作，不会对对象拷贝或包含的数据。

``` python
>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
```

Python 中可变对象以引用的形式进行传递，因此函数也不进行拷贝

### 视图和浅拷贝

不同的数组可以共享相同的数据，view方法会创建一个共享原数据数据的新数组。若数组 A 为数组 B 的 视图，则称 B 为 A 的 base。视图数组中的数据实际保存于原数组(base)中.

``` python
>>> c = a.view()
>>> c is a
False
>>> c.base is a
True
>>> a.flags.owndata
True
>>> c.flags.owndata
False
>>> c.shape = 2,6                      # a的形状并不随之改变
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a的数据也会变
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
```

此外，数组切片返回的也是其视图。

``` python
>>> s = a[:, 1:3]
>>> s[:] = 10	# s[:] 为 s 的视图
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

### 深拷贝

`copy`方法对数组及其数据进行完全的拷贝。

``` python
>>> d = a.copy()
>>> d is a
False
>>> d.base is a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
```

## 广播

Universal functions 对两个数组进行计算时，会对数组的对应元素进行计算。当数组的 shape 不同时，便会进行广播处理。其规则如下：

1. 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐
2. 输出数组的 shape 是输入数组 shape 的各个轴上的最大值
3. 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
4. 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值

## 参考

1. [Quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
2. [NumPy Reference](https://docs.scipy.org/doc/numpy-1.13.0/reference/index.html)
