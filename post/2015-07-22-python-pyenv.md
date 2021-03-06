---
title: Python多版本共存之pyenv
date: 2015-07-22
lastMode: 2015-07-22
author: pzhang
category:
  - 安装
tags:
  - pyenv
  - Python

slug: python-env
---


经常遇到这样的情况：

- 系统自带的Python是2.6，自己需要Python 2.7中的某些特性；
- 系统自带的Python是2.x，自己需要Python 3.x；

此时需要在系统中安装多个Python，但又不能影响系统自带的Python，即需要实现Python的多版本共存。[pyenv](https://github.com/yyuu/pyenv) 就是这样一个Python版本管理器。

## 安装pyenv

``` bash
$ git clone git://github.com/yyuu/pyenv.git ~/.pyenv
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PATH:$PYENV_ROOT/bin"' >> ~/.bashrc
$ echo 'eval "$(pyenv init -)"' >> ~/.bashrc
$ exec $SHELL -l
```


<!--more-->

## 安装Python

### 查看可安装的版本

   $ pyenv install --list

该命令会列出可以用pyenv安装的Python版本，仅列举几个::

    2.7.8   # Python 2最新版本
    3.4.1   # Python 3最新版本
    anaconda-2.0.1  # 支持Python 2.6和2.7
    anaconda3-2.0.1 # 支持Python 3.3和3.4

其中形如 `x.x.x` 这样的只有版本号的为Python官方版本，其他的形如 `xxxxx-x.x.x` 这种既有名称又有版本后的属于“衍生版”或发行版。

### 安装Python的依赖包

在安装Python时需要首先安装其依赖的其他软件包，已知的一些需要预先安装的库如下。

在CentOS/RHEL/Fedora下::

    sudo yum install readline readline-devel readline-static
    sudo yum install openssl openssl-devel openssl-static
    sudo yum install sqlite-devel
    sudo yum install bzip2-devel bzip2-libs

### 安装指定版本

使用如下命令即可安装python 3.4.1：

    $ pyenv install 3.4.1 -v

该命令会从github上下载python的源代码，并解压到/tmp目录下，然后在/tmp中执行编译工作。若依赖包没有安装，则会出现编译错误，需要在安装依赖包后重新执行该命令。

对于科研环境，更推荐安装专为科学计算准备的Anaconda发行版，` pyenv install anaconda-2.1.0` 安装2.x版本，`pyenv install anaconda3-2.1.0` 安装3.x版本；

Anacoda很大，用pyenv下载会比较慢，可以自己到Anaconda官方网站下载，并将下载得到的文件放在 `~/.pyenv/cache` 目录下，则pyenv不会重复下载。

### 更新数据库

安装完成之后需要对数据库进行更新：


    $ pyenv rehash

### 查看当前已安装的python版本


    $ pyenv versions
    * system (set by /home/seisman/.pyenv/version)
    3.4.1

其中的星号表示当前正在使用的是系统自带的python。

### 设置全局的python版本


    $ pyenv global 3.4.1
    $ pyenv versions
    system
    * 3.4.1 (set by /home/seisman/.pyenv/version)

当前全局的python版本已经变成了3.4.1。也可以使用 `pyenv local` 或 `pyenv shell` 临时改变python版本。

### 确认python版本


    $ python
    Python 3.4.1 (default, Sep 10 2014, 17:10:18)
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>

## 使用python

-  输入 `python` 即可使用新版本的python；
-  系统自带的脚本会以 `/usr/bin/python` 的方式直接调用老版本的python，因而不会对系统脚本产生影响；
-  使用 `pip` 安装第三方模块时会安装到 `~/.pyenv/versions/3.4.1` 下，不会和系统模块发生冲突。
-  使用 `pip` 安装模块后，可能需要执行 `pyenv rehash` 更新数据库；

## 更新

更新至最近的pyenv版本，可以用git的pull命令


    $ cd ~/.pyenv
    $ git pull

想要更新至某个特定的pyenv版本，也行

``` bash

    $ cd ~/.pyenv
    $ git fetch
    $ git tag  # 会显示出所有可用的版本号
    v0.1.0
    v0.1.1
    v0.1.2
    v0.2.0
    v0.2.1
    v0.3.0
    v0.4.0
    v0.4.0-20130613
    v0.4.0-20130726
    v0.4.0-20131023
    v0.4.0-20131116
    v0.4.0-20131216
    v0.4.0-20131217
    v0.4.0-20140110
    v0.4.0-20140110.1
    ...
    $ git checkout v0.1.0
```

## 参考

#. https://github.com/yyuu/pyenv
#. http://seisman.info/python-pyenv.html

## 修订历史

- 2015-07-22：初稿取自Seisman；
- 2015-09-01：添加了更新pyenv部分
