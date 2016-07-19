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
    $ sudo yum install gd gd-devel      # PNG support
    $ sudo yum install cairo cairo-devel # cairo support
    
分别为X11 graphics, vplot2gif, TIFF, JPEG, PLplot, OpenGL和ppm支持

### 数值依赖包

    $ sudo yum install blas blas-devel atlas atlas-devel lapack lapack-devel
    $ sudo yum install mpich mpich-devel
    $ sudo yum install fftw fftw-devel
    
### 其他

- Seismic_Unix
- CurveLab, PyCurveLab
- 

## 下载安装


- 稳定版

[Download the source code distribution from Sourceforge](http://sourceforge.net/projects/rsf/files/)


    $ tar jxvf madagascar-1.7.tar.bz2
    $ cd madagascar-1.7
    $ ./configure --preifx=/home/pzhang/seisCode/rsf API=f90,python,matlab
    $ make
    $ make install
    
- 最新版

直接从 GitHub 克隆下来：

    $ git clone https://github.com/ahay/src.git RSF_SRC
    $ cd RSF_SRC
    $ ./configure --preifx=/home/pzhang/seisCode/rsf API=f90,c++,python,matlab
    $ make
    $ make install

## 添加环境变量

    $ echo "source /home/pzhang/rsf/share/madagascar/etc/env.sh" >> ~/.bashrc
    $ source ~/.bashrc
    
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
    

## 待解决的问题

一些相关包安装后，`configure` 时仍然是 NO。列表如下：

- blas
- lapack
- plplot


## 参考

- [Madagascar Installation](http://www.ahay.org/wiki/Installation#Precompiled_binary_packages)
- [Advanced Installation](http://www.ahay.org/wiki/Advanced_Installation#Platform-specific_installation_advice>)


## 修订历史

-   2015-10-26： 初稿
-   2016-07-19： 更新存在的问题




