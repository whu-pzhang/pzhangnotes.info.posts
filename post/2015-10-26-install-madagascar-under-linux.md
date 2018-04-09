---
title: Linux下安装Madagascar
author: pzhang
date: 2015-10-26
lastMod: 2015-10-26
categories:
  - 安装
  - 地震勘探
tags:
  - Linux
  - Madagascar

slug: install-madagascar-on-linux
---


Madagascar 是与Seismic_Unix 以及 SEPlib 差不多的一套东西。


<!--more-->

## 依赖

需要添加 [EPEL](https://fedoraproject.org/wiki/EPEL) 源 和 [Nux Dextop](http://li.nux.ro/repos.html) 源支持。

    $ sudo yum install -y epel-release
    $ sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

### 基础依赖包

    $ sudo yum install gcc libXaw-devel python

值的注意的是，Madagascar只支持`Python2`

### 核心及开发依赖

    $ sudo yum install gcc-c++ gcc-gfortran       # c++和fortran
    $ sudo yum install python-devel swig numpy    # Python API
    $ sudo yum install java     # Java API

要用到MATLAB API的话，自然得安装MATLAB

### 图形和可视化

    $ sudo yum install libtiff-devel libjpeg-turbo-devel
    $ sudo yum install freeglut freeglut-devel  # opengl
    $ sudo yum install netpbm netpbm-devel      # ppm support
    $ sudo yum install gd gd-devel              # PNG support
    $ sudo yum install plplot plplot-devel      # plplot support
    $ sudo yum install ffmpeg ffmpeg-devel      # need nux-dextop support
    $ sudo yum install cairo cairo-devel        # cairo support

分别为X11 graphics, vplot2gif, TIFF, JPEG, PLplot, OpenGL和ppm支持

### 数值依赖包

    $ sudo yum install blas blas-devel atlas atlas-devel lapack lapack-devel
    $ sudo yum install fftw fftw-devel
    $ sudo yum install suitesparse suitesparse-devel    # 稀疏矩阵库

`mpi`环境的安装可以参看 [CentOS7安装及配置](linux-environment-for-seismology-research.html) 中的并行计算部分

### 其他

- Seismic_Unix
- CurveLab, PyCurveLab
-

## 安装

### 添加环境变量

将以下语句加入 `~/.bashrc` 或 `~/.zshrc` 中

``` bash
export RSFROOT=/opt/rsf # 安装位置
source ${RSFROOT}/share/madagascar/etc/env.sh
```

### 编译安装

- 稳定版

[Download the source code distribution from Sourceforge](http://sourceforge.net/projects/rsf/files/)


    $ tar jxvf madagascar-1.7.tar.bz2
    $ cd madagascar-1.7
    $ ./configure --preifx=/opt/rsf API=f90,python,matlab
    $ make
    $ sudo make install

- 最新版

直接从 GitHub 克隆下来：

    $ git clone https://github.com/ahay/src.git RSF_SRC
    $ cd RSF_SRC
    $ ./configure --preifx=/opt/rsf API=f90,c++,python,matlab
    $ make
    $ sudo make install


## 卸载

清除所有中间文件和安装文件

    $ scons -c install

或者在源码文件夹中

    $ make distclean

## 测试

    $ sfspike n1=1000 k1=300 | sfbandpass fhi=2 phase=y | \
        sfwiggle clip=0.02 title="Welcome to Madagascar" | sfpen

不出错且出现如下图形即安装成功

![](http://www.ahay.org/wiki2015/images/3/35/Filter.png)


## 出现的问题及解决方案

### BLAS

即使安装了`blas blas-devel lapack lapack-devel`，在`./configure`时还有如下提示：

    checking for BLAS ... no
    checking for LAPACK ... no

查看 `config.log` 后发现其利用的是 `cblas`，而在CentOS7中`cblas`为 `atlas` 包的
一部分，这样在调用`cblas`中的函数时，默认情况下在编译时需加上`-I/usr/include`来
包含`cblas.h`，链接时，则需加上`-L/usr/lib64/atlas -lsatlas`或
者`-L/usr/lib64/atlas -ltatlas`。其中,`s`表示`single`;`t`表示`multi-threaded`。

有了如上信息，我们可以有如下两种解决方案：

#### 自己创建一个cblas库文件

这种方法是在不修改Madagascar源文件的前提下进行的。

第一步，将`satlas`或`tatlas`库软链接成`cblas`库：

    $ sudo ln -s /usr/lib64/atlas/libsatlas.so /usr/lib64/libcblas.so

第二步，`./configure`时侯，加上`BLAS=cblas`

    $ ./configure --preifx=/home/pzhang/src.import/rsf API=f90,c++,python,matlab BLAS=cblas

#### 修改`framework/configure.py`

打开`framework/configure.py`文件，找到`check_all(context)`函数，可以发现其调
用`blas(context)`来对`BLAS`库进行检查，再定位到`blas(context)`函数，修改以下
语句：

``` python
# Line 1014
LIBS.append('f77blas')
LIBS.append('cblas')
LIBS.append('atlas')

# Line 1066
mylibs = ['f77blas','cblas','atlas']
```
为：
``` python
#LIBS.append('f77blas') # 因为CentOS7系统上blas库都集成到atlas中了
#LIBS.append('cblas')
LIBS.append('satlas')

#mylibs = ['f77blas','cblas','atlas']
mylibs = ['satlas']
```

上面两种方法任取一种即可！

### plplot

在安装了`plplot`和`plplot-devel`后，`configure`时`plplot`仍然是 `no`。

同样的，我们查看`config.log`可以发现是在链接时找不到名为`ltdl`的库。

    $ ls /usr/lib64 | grep ltdl
    libltdl.so.7
    libltdl.so.7.3.0

也就是说，没有名为`libltdl.so` 的动态库文件，要解决的话也很简单，找不到库文件我
们自己造一个：

    $ sudo ln -s /usr/lib64/libltdl.so.7 /usr/lib64/libltdl.so


**PS: 类似其他的`checking NO` 的问题，都可以在`config.log`中找到原因！**

## 其他可能出现的问题

在研究所里的曙光超算(`CentOS5.5 Final, gcc-4.1.2, scons-2.3.1,python2.4`)上
安装时出现的问题。

在该环境下，直接`./configure`时，会报如下错误：
```
checking for C compiler ... (cached) gcc
checking if gcc works ... (cached) no

  Fatal missing dependency
------------------------
```

Google后发现这个报错是由`scons-2.3.1`对`Python-2.6`之前版本存在兼容性问题
而导致的。

有如下三种解决办法：

1. 升级`Python`到较新的版本(2.7)

2. 降级`Scons`(2.3.0)

3. 编辑 `/usr/local/lib/scons-2.3.1/SCons/Node/__init__.py` 文件，Line 1004
``` python
return list(chain.from_iterable(filter(None, [self.sources, self.depends, self.implicit])))
```
替代为:
``` python
if self.implicit is None:
    return self.sources + self.depends
else:
    return self.sources + self.depends + self.implicit
```




## 参考

- [Madagascar Installation](http://www.ahay.org/wiki/Installation#Precompiled_binary_packages)
- [Advanced Installation](http://www.ahay.org/wiki/Advanced_Installation#Platform-specific_installation_advice>)
- [How can I install Madagascar when the gcc version is 4.1.2](https://groups.google.com/forum/#!topic/osdeve_mirror_geophysics_rsf-user/AwnyyKjJ2SI)


## 修订历史

-   2015-10-26： 初稿
-   2016-07-19： 更新存在的问题
-   2016-10-07： 更新问题的解决方案
-   2016-11-10： 添加BLAS库找不到的另一种解决方案
-   2016-11-14： 添加在曙光超算上出现的问题
