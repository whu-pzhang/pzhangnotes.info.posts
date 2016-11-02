---
title: Linux下安装Madagascar
author: pzhang
date: 2015-10-26
category: Exploration Seismology
tags: [Linux, 安装]
---


Madagascar 是与Seismic_Unix 以及 SEPlib 差不多的一套东西。


<!--more-->

## 依赖

需要添加 [EPEL](https://fedoraproject.org/wiki/EPEL) 源 和 [Nux Dextop](http://li.nux.ro/repos.html) 源支持。

    $ sudo yum install -y epel-release
    $ sudo rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

### 基础依赖包

    $ sudo yum install gcc libXaw-devel python

值的注意的是，Madagascar只支持Python2

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

![](/images/2015102600.png)


## 出现的问题

### blas

即使安装了`blas blas-devel lapack lapack-devel`，在`./configure`时还有如下提示：

    checking for BLAS ... no
    checking for LAPACK ... no

查看 `config.log` 后发现其利用的是 `cblas`，而在CentOS7中`cblas`为 `atlas` 包的一部分，
这样在调用`cblas`中的函数时，默认情况下在编译时需加上`-I/usr/include`来包含`cblas.h`，
链接时，则需加上`-L/usr/lib64/atlas -lsatlas`或者`-L/usr/lib64/atlas -ltatlas`。
其中,`s`表示`single`;`t`表示`multi-threaded`。

有了如上信息，我们可以有如下解决方案：

第一步，将`satlas`或`tatlas`库软链接成`cblas`库：

    $ sudo ln -s /usr/lib64/atlas/libsatlas.so /usr/lib64/libcblas.so

第二步，`./configure`时侯，加上`BLAS=cblas`即可检测到`BLAS`和`LAPACK`了

    $ ./configure --preifx=/home/pzhang/src.import/rsf API=f90,c++,python,matlab BLAS=cblas

### plplot

在安装了`plplot`和`plplot-devel`后，`configure`时`plplot`仍然是 `no`。

同样的，我们查看`config.log`可以发现是在链接时找不到名为`ltdl`的库。

    $ ls /usr/lib64 | grep ltdl
    libltdl.so.7
    libltdl.so.7.3.0

也就是说，没有名为`libltdl.so` 的动态库文件，要解决的话也很简单，找不到库文件我们自己
造一个：

    $ sudo ln -s /usr/lib64/libltdl.so.7 /usr/lib64/libltdl.so


**PS: 类似其他的`checking NO` 的问题，都可以在`config.log`中找到原因！**

## 参考

- [Madagascar Installation](http://www.ahay.org/wiki/Installation#Precompiled_binary_packages)
- [Advanced Installation](http://www.ahay.org/wiki/Advanced_Installation#Platform-specific_installation_advice>)


## 修订历史

-   2015-10-26： 初稿
-   2016-07-19： 更新存在的问题
-   2016-10=07： 更新问题的解决方案
