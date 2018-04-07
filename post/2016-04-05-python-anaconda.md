---
title: Python科学计算环境Anaconda
date: 2016-04-05
author: pzhang
categories:
  - Programming
tags:
  - Python
  - Anaconda
---


本文记录Python科学计算发行版Anaconda的介绍，安装以及后续的一些应用。

## Anaconda 特性

主页： https://www.continuum.io/

- 包含了众多流行的科学、教学、工程和数据分析的Python包
- 完全开源、免费
- 对于学术用途可以申请免费的加速 icense
- 全平台支持： Linux、Windows、Mac
- 支持Python2.6、2.7、3.4、3.5

<!--more-->

## 安装

### 安装pyenv

利用 [pyenv](https://github.com/yyuu/pyenv) 来对多个版本的python进行管理。

首先安装好pyenv。具体见 [Python多版本共存之pyenv](/python-pyenv.html)

### 安装Anaconda

pyenv支持很多版本的anaconda

python2.7

    $ pyenv install anaconda-2.0.1 -v

或者python3

    $ pyenv install anaconda3-2.5.1 -v

### 申请免费的学术License

对于由教育邮箱的学生或者教职工，可以申请免费的学术License，实现计算过程的加速。

申请地址：https://www.continuum.io/anaconda-academic-subscriptions-available

注册一个帐号，然后在 My settings-->Add ons里就可以找到额外的加速License了，
下载下来后是几个txt后缀的文本，将其移动到 `$HOME/.continuum` 目录下
（若不存在该目录，则新建一个）

这些额外的包包括：

- MKL Optimizations

    Boost the speed of NumPy, SciPy, NumExpr, and scikit-learn through Intel's Math Kernel Library (MKL)
- IOPro

    Fast, memory-efficient Python interface for databases, NoSQL stores, Amazon S3, and large data files.
- Anaconda Accelerate

    Fast Python for GPUs and multi-core with NumbaPro and MKL Optimizations.

### 安装额外加速包

    $ conda update conda
    $ conda install accelerate
    $ conda update numbapro
    $ conda install iopro


## 模块管理

Anaconda自带了conda用于模块的管理。

    # 安装模块
    $ conda install scipy
    # 更新模块
    $ conda update scipy
    # 更新所有模块
    $ conda update --all


也可以利用conda来对Anaconda进行升级

    $ conda update anaconda



## 参考

1. [ADVANCED INSTALL INSTRUCTIONS](https://docs.continuum.io/advanced-installation)
