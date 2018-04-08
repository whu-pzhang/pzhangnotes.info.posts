---
title: MPI学习 - MPI配置
author: pzhang
date: 2018-04-08
lastMod: 2018-04-08
categories:
  - Programming
tags:
  - MPI
  - 高性能计算

draft: true
slug: mpi-intro
---

作为MPI系列学习笔记的开始，本文介绍如何在服务器上配置MPI环境。

<!--more-->

## MPI 安装

安装部分略过不谈。自行下载源码编译或者从Linux发行版仓库安装均可。

安装完成后

```shell
$ mpicc --showme
llvm-clang -I/usr/local/Cellar/open-mpi/3.0.0_2/include -L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/3.0.0_2/lib -lmpi
```
## 服务器上的配置

MPI通常是运行在多节点服务器上，单机并行直接用OpenMP可能更为方便。
