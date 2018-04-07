---
title: 用ctypes调用C函数
date: 2017-05-11
author: pzhang
categories:
  - Programming
tags:
  - Python
  - c
slug: python-ctypes
---



## 前言

现在已有一些C语言函数被编译成共享库，我们想从纯Python中直接调用这些函数，而不必额外编写C代码或者使用第三方的扩展工具。

对于这样的需求，使用Python标准库中的`ctypes`模块来实现是非常容易的。

<!--more-->

要使用ctypes，必须确保要访问的C代码已经被编译成与Python解释器相兼容（即，采用相同的体系结构、字长、编译器等）的共享库。

## C源码

现有如下的C代码：

```c
#include <math.h>
#include <stdbool.h>
#include "sample.h"

// 计算最大公约数
int gcd(int x, int y)
{
    int g = y;
    while (x > 0) {
        g = x;
        x = y % x;
        y = g;
    }
    return g;
}

// 检查（x0,y0）是否在Mandelbort集合中
bool in_mandel(double x0, double y0, int n)
{
    double x=0, y=0, xtemp;
    while (n > 0) {
        xtemp = x*x -y*y + y0;
        y = 2*x*y + y0;
        x = xtemp;
        n -= 1;
        if (x*x + y*y > 4) return false;
    }
    return true;
}

// 两数相除
int divide(int a, int b, int *remainder)
{
    int quot = a / b;
    *remainder = a % b;
    return quot;
}


// 计算数组平均值
double avg(double *a, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += a[i];

    return sum / n;
}

// C 结构体
typedef struct point {
    double x, y;
} Point;

double distance(Point *p1, Point *p2)
{
    return hypot(p1->x - p2->x, p1->y - p2->y);
}
```

这份代码中包含了C语言中不同的特性。其中`gcd`和`in_mandel`为不同参数类型的简单函数。`divide`则是C语言中返回多个值的例子，其中一个以指针的形式返回。`avg`函数遍历了数组且做了数据转换。`distance`则涉及到C语言结构体类型。

将上面的代码写入了一个名叫“sample.c”的文件中， 然后它们的声明写入名为“sample.h”的头文件中。

我们首先将上述源码编译成共享库：

``` bash
gcc -std=c99 -Wall -fPIC -shared sample.c -o libsample.so
```

## 利用ctype访问C代码

要访问我们编译好的共享库`libsample.so`，需要先构建一个Python模块来包装它，示例如下：

``` python
#!/usr/bin/env python3
# coding: utf-8

import os
import ctypes

# 定位到共享库文件
_file = 'libsample.so'
_path = os.path.join(*(os.path.split(__file__)[:-1] + (_file,)))
_mod = ctypes.cdll.LoadLibrary(_path)

# int gcd(int, int)
gcd = _mod.gcd
gcd.argtypes = (ctypes.c_int, ctypes.c_int)
gcd.restype = ctypes.c_int

# bool in_mandel(double, double, int)
in_mandel = _mod.in_mandel
in_mandel.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_int)
in_mandel.restype = ctypes.c_bool

# int divide(int, int, int *)
_divide = _mod.divide
_divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_divide.restype = ctypes.c_int


def divide(x, y):
    rem = ctypes.c_int()
    quot = _divide(x, y, rem)

    return quot, rem.value

# double avg(doubel *, int n)
# 定义一个类来处理 "doubel *" 参数
class DoubleArrayType:

    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, ctypes.Array):
            return param
        else:
            raise TypeError("Can't convert %s" % typename)

    # 从列表和元组转换
    def from_list(self, param):
        val = ((ctypes.c_double) * len(param))(*param)
        return val

    from_tuple = from_list

    # 从array.array对象转换
    def from_array(self, param):
        if param.typecode != 'd':
            raise TypeError("must be an array of doubles")
        ptr, _ = param.buffer_info()
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))

    # 从nump数组对象转换
    def from_ndarray(self, param):
        if param.dtype.name != 'float64':
            raise TypeError("The dtype of array must be float64")
        return param.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


DoubleArray = DoubleArrayType()
_avg = _mod.avg
_avg.argtypes = (DoubleArray, ctypes.c_int)
_avg.restype = ctypes.c_double


def avg(values):
    return _avg(values, len(values))


# struct Point {}
class Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]

# double distance(Point *, Point *)
distance = _mod.distance
distance.argtypes = (ctypes.POINTER(Point), ctypes.POINTER(Point))
distance.restype = ctypes.c_double
```

如果一切正常，就可以加载并使用里面定义的C函数了：


```python
>>> import sample
>>> sample.gcd(35, 42), sample.in_mandel(0, 0, 500), sample.divide(42, 8)
```
    (7, True, (5, 2))

```python
>>> import array
>>> import numpy as np

>>> a = list(range(100))
>>> b = array.array('d', a)
>>> c = np.array(a, dtype=np.double)  # dtype需为int64

>>> sample.avg(a), sample.avg(b), sample.avg(c)
```
    (49.5, 49.5, 49.5)


```python
>>> p1 = sample.Point(1, 2)
>>> p2 = sample.Point(4, 5)
>>> sample.distance(p1, p2)
```
    4.242640687119285

测试下速度：


```python
>>> a = np.random.rand(100)
>>> %timeit a.mean()
```
    4.28 µs ± 20.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

```python
>>> %timeit sample.avg(a)
```
    6.45 µs ± 72 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

可以看到，调用C函数的速度随比不上NumPy，但已经很快了！

## 细节剖析

### 搜索路径

用ctypes来调用自己编译的C程序，需要把共享库放在驱动模块可以找到的地方，上述例子中将共享库文件与所支撑的Python文件放在同一目录下：`sample.py`通过`__file__`获得当前目录，然后在相同目录下构建一个指向`libsample.so`的路径。

如果想将C程序共享库安装到别的地方，则需要调整路径。若C共享库已经作为标准库安装在机器上了，那么可以直接使用`ctypes.util.find_library()`函数。


```python
>>> from ctypes.util import find_library
>>> find_library('m')
```
    'libm.so.6'

```python
>>> find_library('fftw3')
```
    'libfftw3.so.3'

一旦ctypes无法找到C共享库则无法工作。

知道C共享库的位置后，就可以使用`ctypes.cdll.LoadLibrary()`函数来加载。上述例子中，`_path` 为指向共享库的完整路径，用以下语句来加载：

```python
_mod = ctypes.cdll.LoadLibraay(_path)
```

### 源码说明

C共享库加载完成后，我们还需要将Python的数据类型转换成C语言可以直接使用的参数类型。

对不同的C函数，需要不同的处理方案，这里将其分为三类。

#### 常规函数

这里的常规函数指的是像`int gcd(int, int);`和`bool in_mandel(double, double, int)`这样参数中不含指针，返回值单一的函数。我们摘录一段代码来看：

``` python
# int gcd(int, int)
gcd = _mod.gcd
gcd.argtypes = (ctypes.c_int, ctypes.c_int)
gcd.restype = ctypes.c_int
```

其中，`.argtypes` 属性为一个包含函数输入参数的元组，而`.restype`代表返回值类型。`ctypes`中定义了很多类型对象：`c_int`，`c_float`，`c_uint`等，用来表示常见的C数据类型。

用Python调用时，需要传递正确的参数类型并对数据作正确的转换，代码才能正常工作。
因此，类型签名的绑定至关重要。

对于常规C函数，只需要对数据类型作相应的转换就行。

#### 多返回值的函数

由于C语言不支持返回多个值，此类需求通常以指针的形式实现。例子中的`int divide(int, int, int *);`函数便是这类函数。

对于这类函数，我们不能和常规函数一样处理：


```python
>>> import ctypes
>>> _mod = ctypes.cdll.LoadLibrary('./libsample.so')
>>> divide = _mod.divide
>>> divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
>>> x = 0
>>> divide(10, 3, x)
```

    ---------------------------------------------------------------------------

    ArgumentError                             Traceback (most recent call last)

    <ipython-input-20-f3752d915faa> in <module>()
          4 divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
          5 x = 0
    ----> 6 divide(10, 3, x)

    ArgumentError: argument 3: <class 'TypeError'>: expected LP_c_int instance instead of int

即使这样行的通，也违反了Python中整数不可变的原则，可能会导致整个进程卡死。

对于涉及指针的参数，必须构建一个兼容的`ctypes`对象才行


```python
>>> x = ctypes.c_int()
>>> divide(10, 3, x)
```
    3

```python
>>> x.value
```
    1

这里，我们创建了一个`ctypes.c_int`对象，并把它作为指针对象传递给函数，与普通的Python整数不同，`ctypes.c_int`对象是可变的，可以根据需要通过`.value`属性来获取或者修改值。

对于那些不Pythonic的C函数调用，通常需要写一个包装函数。这里，`divide()`函数通过元组返回两个结果。


```python
# int divide(int, int, int *)
_divide = _mod.divide
_divide.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_divide.restype = ctypes.c_int


def divide(x, y):
    rem = ctypes.c_int()
    quot = _divide(x, y, rem)

    return quot, rem.value
```

#### 处理数组的函数

对于`avg()`函数来说，C代码需要接收一个指针和一个表示数组长度的整型。从Python的角度来看，我们需要考虑以下问题：数组是什么？是列表还是元组？是`array`模块中的`array`对象还是`numpy`中的`ndarray`对象？ 实际上，Python中的数组有多种形式，我们这里就需要考虑这多种形式。

`DoubleArrayType`类展示了如何处理这种情况。在这个类中，`from_param()`方法的作用就是接收一个单独的 参数并将其范围缩小为一个兼容的`ctypes`对象，在本例中就是指向`ctypes.c_double`的指针。在`from_param()`方法中，参数的类型名被提取出来并被分发给一个更具体的方法中，如果`typename`为列表，就调用`from_list()`方法。

对于列表和元组，`from_list()` 方法将其转换为一个 `ctypes` 的数组对象。

对于数组对象，`from_array()` 提取底层的内存指针并将其转换为一个 `ctypes` 指针对象。

`from_ndarray()` 演示了对于 `numpy` 数组的转换操作。

这样，通过定义 `DoubleArrayType` 类并在 avg() 类型签名中使用它， 那么这个函数就能接受多个不同的类数组输入了。

#### 结构体

对于C结构体，只需要定义一个类，在其中包含适当的字段和类型：

```python
# struct Point {}
class Point(ctypes.Structure):
    _fields_ = [('x', ctypes.c_double),
                ('y', ctypes.c_double)]
```

类被定义后，你就可以在类型签名中或者是需要实例化结构体的代码中使用它。

## 参考

1. [Python Cookbook](http://chimera.labs.oreilly.com/books/1230000000393)
2. [ctypes](https://docs.python.org/3/library/ctypes.html#module-ctypes)
