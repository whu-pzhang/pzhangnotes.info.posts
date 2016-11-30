---
title: sfspray和sfstack
date: 2016-09-12 18:57:10
author: pzhang
categories: Exploration Seismology
tags: [Madagascar, program of week]
---

## 前言

spray和stack是一对adjoint算子。

<!--more-->

## sfstack



## sfspray

### 文档说明

spray 的意思是 “喷射，散射”。
对于一维来说，就是取一个值，然后将其分散到空间中的很多值上。

`sfspray` 的文档如是说： 在指定轴上通过重复来扩展数据体。其输入将会获得一个额外的数据轴。

关键参数只有一个 `axis=`， 用来指定扩展哪个维度。

直接来看例子,先创建一个二维的数据体：

    $ sfmath n1=5 n2=2 output=x1+x2 > test.rsf
    $ sfin test.rsf
    test.rsf:
    in="/var/tmp/test.rsf@"
    esize=4 type=float form=native
    n1=5           d1=1           o1=0          
    n2=2           d2=1           o2=0          
	10 elements 40 bytes
    $ < test.rsf sfdisfil
    0:             0            1            2            3            4
    5:             1            2            3            4            5

然后，我们利用 `sfspray` 来扩展第二个数据轴。

    $ < test.rsf sfspray axis=2 n=3 > test2.rsf
    $ sfin test2.rsf
    test2.rsf:
    in="/var/tmp/test2.rsf@"
    esize=4 type=float form=native
    n1=5           d1=1           o1=0          
    n2=3           d2=1           o2=0          
    n3=2           d3=1           o3=0          
    n4=1           d4=?           o4=?          
	30 elements 120 bytes
    $ < test2.rsf sfdisfil
      0:             0            1            2            3            4
      5:             0            1            2            3            4
     10:             0            1            2            3            4
     15:             1            2            3            4            5
     20:             1            2            3            4            5
     25:             1            2            3            4            5

按照文档的说法，输出应该是三维的数据，但是我们的输出里还有个 `n4=1` 是什么鬼？！

目的达到了：数据沿着第二轴重复，将数据体扩展到了三维。

    $ < test.rsf sfspray axis=3 n=2 > test3.rsf
    $ sfin test3.rsf
    test3.rsf:
    in="/var/tmp/test3.rsf@"
    esize=4 type=float form=native
    n1=5           d1=1           o1=0          
    n2=2           d2=1           o2=0          
    n3=2           d3=?           o3=?          
    n4=1           d4=?           o4=?          
	20 elements 80 bytes
    $ < test3.rsf sfdisfil
      0:             0            1            2            3            4
      5:             1            2            3            4            5
     10:             0            1            2            3            4
     15:             1            2            3            4            5
此时的输出数据沿着第三个轴重复了。


### 个人理解及应用

前面基本是按照官方文档翻过来的，感觉有点晦涩。
按照我自己的理解，`sfspray` 其实就是复制粘贴！ 将`axis`值前面的部分复制，然后粘贴
`n-1`遍。原数据体`axis`后的部分依次后移一个轴。

举个例子：就以上面的 `test.rsf` 来说，其 `n1=5 n2=2`，那么 `sfspray axis=2 n=3`
就是将`axis=2`之前的`n1`轴的数据复制粘贴2遍，再将`n2=2`变成`n3=2`即可。

目前想到的一个应用： 去直达波！

## 个人笔记

看源码学到了三个比较有用的函数：
``` c
off_t sf_shiftdim(sf_file in, sf_file out, int axis);
/* 在输入文件已有轴的基础上，将其移动一维
Input:
    in:     a pointer to the input file structure (sf file).
    out:    a pointer to the output file structure (sf file).
    axis:   the axis after which the grid is to be shifted (sf file).
Output:
    n3:     shift轴 之后的文件大小(不是字节大小，而是维度的乘积)
        例如n1=5 n2=2的输入文件，n3=sf_shiftdim(in, out, 2)后， 输出文件
        out中n1=5 n3=2 */

void sf_fileflush (sf_file file, sf_file src);
/* 将参数从源文件输出到目标文件，设定目标文件的格式并且准备好写二进制数据
    file:   pointer to the output file (sf file).
    src:    a pointer to the input file structure (sf file).*/

void sf_setform (sf_file file, sf_dataform form);
/* 设置rsf文件的格式，可选有： SF_ASCII, SF_XDR, SF_NATIVE
    file:   a pointer to the file structure whose form is to be set (sf file)
    form:   the type to be set. Must be of type sf datatype, e.g. SF ASCII.*/
```
