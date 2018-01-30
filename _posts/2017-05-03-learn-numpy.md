---
title: NumPy学习笔记
tags:
  - numpy
author: pzhang
categories: Programming
date: 2017-05-03 00:00:00
---


[NumPy](http://www.numpy.org/) 为 Python下科学计算的基础包。

## 简介

NumPy 的出现主要是弥补 Python 数值计算的不足。为 Python 科学计算提供了强大的支撑。

NumPy 主要提供了两种对象：ndarray(N-dimensional array)和 ufunc(Unversal function)。

<!--more-->

## 基础篇

一个 ndarray (以下称为数组)是包含同种元素类型的多维数组容器。在 NumPy 中维度(dimensions) 叫做axes，轴的个数称为 rank。

例如，在 3D 空间一个点的坐标`[1, 2, 3]` 为一个rank为1的数组，因为它只有一个axes，轴的长度为3。又例如，在以下例子中，数组的rank为2（它有两个维度），第一维长为2，第二维长度为3.

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
>>> print(type(a))
<class 'numpy.ndarray'>
>>> print(a.shape)
(3,)
>>> b = np.array([[1, 2, 3],[4, 5, 6]])
>>> print(b.shape)
(2, 3)
```

NumPy 提供了很多函数来的创建数组：

```Python
>>> a = np.zeros((2,2))
>>> print(a)
[[ 0.  0.]
 [ 0.  0.]]
>>> b = np.ones((1,2))
>>> print(b)
[[ 1.  1.]]
>>> c = np.full((2,2), 3)
>>> print(c)
[[3 3]
 [3 3]]
>>> d = np.eye(2)
>>> print(d)
[[ 1.  0.]
 [ 0.  1.]]
>>> e = np.random.random((2,2))
>>> print(e)
[[ 0.76012262  0.94708717]
 [ 0.20903943  0.30772447]]
>>> f = np.arange(5)
>>> print(f)
[0 1 2 3 4]
>>> g = np.linspace(1,10,10)
>>> g
array([  1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])
```

由于浮点数有限精度的限制，通常利用` linspace` 创建浮点数组更好。

除了上述函数以外，还有`zeros_like`, `ones_like`, `empty_like` , `fromfunction`和 `fromfile`等函数也能来创建数组。详见[Array creation routines](https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.array-creation.html#routines-array-creation).

###基本操作

####索引

NumPy 有多种方式进行索引。

##### slice indexing

同lists，tuple一样，指定每一维的取值范围即可。

```python
>>> a = np.arange(1,13).reshape(3,4)
>>> print(a)
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
>>> b = a[:2, 1:3] # 取包含前2行以及第1，2列的子数组
>>> print(b)
[[2 3]
 [6 7]]
>>> print(a[0,1])
2
>>> b[0, 0] = 88 # 切片数组是原数组的view，改变其值也会改变原数组
>>> print(a[0,1])
88
```

也可以同时使用integer indexing和slice indexing进行索引，但是这样产生的数组的rank比原数组小。

``` python
>>> a = np.arange(1,13).reshape(3,4)
>>> row_r1 = a[1,:]
>>> row_r2 = a[1:2, :]
>>> print(row_r1, row_r1.shape)
[5 6 7 8] (4,)
>>> print(row_r2, row_r2.shape)
[[5 6 7 8]] (1, 4)

>>> col_r1 = a[:,1]
>>> col_r2 = a[:,1:2]
>>> print(col_r1, col_r1.shape)
[88  6 10] (3,)
>>> print(col_r2, col_r2.shape)
[[88]
 [ 6]
 [10]] (3, 1)
```



常规的索引和切片与 Python 中列表和元组的方式是一样的。不再赘述。这里主要讲一下 fancy indexing 和 boolean array indexing。

##### fancy indexing

利用整数列表或数组来进行索引！

``` python
>>> a = np.arange(12).reshape(4,3)
>>> print(a)
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
>>> b = np.array([0,2,0,1]) # 索引数组
>>> print(a[np.arange(4), b])
[ 0  5  6 10]
>>> print(a[[0,2,0,1]])	# 利用 Python 列表进行索引
[[0 1 2]
 [6 7 8]
 [0 1 2]
 [3 4 5]]
>>> a[np.arange(4), b] += 10 # 更改选定元素的值
>>> print(a)
[[10  1  2]
 [ 3  4 15]
 [16  7  8]
 [ 9 20 11]]
>>> arr[[-3, -5, -1]]  # 负数则从末尾行开始选取
array([[ 5.,  5.,  5.,  5.],
       [ 3.,  3.,  3.,  3.],
       [ 7.,  7.,  7.,  7.]])
```

integer indexing 返回的是原数组的copy，不是view

``` python
>>> c = a[np.arange(4), b]
>>> print(a)
[[10  1  2]
 [ 3  4 15]
 [16  7  8]
 [ 9 20 11]]
>>> print(c)
[110 115 116 120]
>>> c += 100
>>> print(c)
[110, 115, 116, 120]
>>> print(a)
[[10  1  2]
 [ 3  4 15]
 [16  7  8]
 [ 9 20 11]]
```

##### Boolean array indexing

布尔型索引通常用于选择数组中满足一定条件的元素。

``` python
>>> arr = np.arange(6).reshape(3, 2)
>>> arr
array([[0, 1],
       [2, 3],
       [4, 5]])
>>> bool_idx = (arr > 2)
>>> bool_idx
array([[False, False],
       [False,  True],
       [ True,  True]], dtype=bool)
>>> arr[bool_idx]
array([3, 4, 5])
>>> arr[arr > 2]  # 通常的写法
array([3, 4, 5])
```

**布尔型索引返回的 view！**

更多关于数组索引的可以参考[array indexing](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)。

#### 维度操作

数组的`shape`属性给出了每个轴(axis)的元素个数。数组的 shape 可以通过函数来改变。



 值得注意的是，上述三个命令都是返回一个视图，对数据的修改都会直接影响原数组。

#### 数学运算

 数组的算术运算都是对数组元素逐个进行的(elmentwise)。这种算术运算称为 Universal function。

``` python
>>> x = np.array([[1,2],[3,4]])
>>> y = np.array([[5,6],[7,8]])
>>> print(x + y) # Elementwise sum，等价于 np.add(x, y)
[[ 6  8]
 [10 12]]
>>> print(x - y) # Elementwise subtract 等价于 np.subtract(x, y)
[[-4 -4]
 [-4 -4]]
>>> print(x * y) # Elementwise product 等价于 np.multiply(x, y)
[[ 5 12]
 [21 32]]
>>> print(x / y) # Elementwise divide 等价于 np.divide(x, y)
[[ 0.2         0.33333333]
 [ 0.42857143  0.5       ]]
>>> print(np.sqrt(x)) # Elementwise square root
[[ 1.          1.41421356]
 [ 1.73205081  2.        ]]
```

NumPy中使用 `np.dot()` 来进行矩阵乘法。从2015年(Python 3.5 & NumPy 1.10) 引入了新的更为方便的运算符 `@`来表示矩阵乘法，可用来代替 `np.dot()`，具体见[PEP 465](https://docs.python.org/3.6/whatsnew/3.5.html#pep-465-a-dedicated-infix-operator-for-matrix-multiplication)。

``` python
>>> x = np.array([[1,2],[3,4]])
>>> y = np.array([[5,6],[7,8]])
>>> 
>>> v = np.array([9,10])
>>> w = np.array([11, 12])
>>> 
>>> print(v @ w) # 向量内积。等价于 np.dot(v,w)
219
>>> print(x @ v) # 矩阵乘向量，右乘作列向量
[29 67]
>>> print(v @ x) # 左乘作行向量
[39 58]
>>> print(x @ y) # 矩阵乘矩阵
[[19 22]
 [43 50]]
```

NumPy还提供了许多有用的函数，如`np.sum()`

``` python
>>> x = np.array([[1, 2], [3, 4]])
>>> print(x.sum()) # 所有元素之和
10
>>> print(np.sum(x, axis=0))  # 计算每列元素之和
[4 6]
>>> print(np.sum(x, axis=1)) # 计算每行元素之和
[3 7]
```

 所有NumPy提供的数学运算函数可见[Mathematical functions](https://docs.scipy.org/doc/numpy/reference/routines.math.html)。

除了数学运算之外，还经常用到改变数组shape的函数，常见的有转置，展平等。

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
更多见[array manipulation](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)。


## 广播

Broadcasting 是NumPy中一种十分高效的机制，允许不同shape的数组之间进行运算。经常遇到的情形是，我们有一个较小的数组和一个较大数组，而我们希望不断地用小数组对大数组进行一些运算。

例如：我们希望给矩阵的每一行加上一个向量。你可能会这么做：

``` python
>>> x = np.arange(1, 13).reshape(4, 3)
>>> v = np.array([1, 0, 1])
>>> y = np.empty_like(x) 
# 矩阵每行加上一个向量
>>> for i in range(4):
...     y[i,:] = x[i, :] + v
    
>>> print(y)
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

这样做结果是正确的，但当数组`x`很大时，这样包含`for`循环的计算效率很低。为了避免循环，我们可以先构建一个与`x`数组维度一致，每一行都是向量`v`的数组`vv`，然后将`vv`于`x`相加(elementwise sum)即可。

```python
>>> vv = np.tile(v, (4, 1)) # 构建数组vv
>>> print(vv)
[[1 0 1]
 [1 0 1]
 [1 0 1]
 [1 0 1]]
>>> z = x + vv # elementwise sum
>>> print(z)
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

NumPy 广播的原理类似，不过不需要显式地构建数组 `vv`。

``` python
>>> y = x + v # 利用广播机制相加
>>> print(y)
[[ 2  2  4]
 [ 5  5  7]
 [ 8  8 10]
 [11 11 13]]
```

尽管 `x` 的shape 为 `(4,3)` 而 `v` 的shape为 `(3,)` ，`y = x + v` 仍然有效。

Universal functions 对两个数组进行计算时，会对数组的对应元素进行计算。当数组的 shape 不同时，便会进行广播处理。其规则如下：

1. 让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分都通过在前面加1补齐
2. 输出数组的 shape 是输入数组 shape 的各个轴上的最大值
3. 如果输入数组的某个轴和输出数组的对应轴的长度相同或者其长度为1时，这个数组能够用来计算，否则出错
4. 当输入数组的某个轴的长度为1时，沿着此轴运算时都用此轴上的第一组值




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

## 参考

1. [Quickstart tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
2. [NumPy Reference](https://docs.scipy.org/doc/numpy-1.13.0/reference/index.html)
