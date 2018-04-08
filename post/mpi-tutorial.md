---
title: MPI 学习笔记
author: pzhang
date: 2018-04-08T13:25:47+08:00
lastMod: 2018-04-08T13:25:47+08:00
categories:
  - Programming
tags:
  - MPI
  - 高性能计算

draft: true
slug: mpi-tutorial
---

Meaasge Passing Interface(MPI)为消息传递接口标准，其目标是为消息传递建立一个便携，高效和
灵活的标准，它将广泛用于编写消息传递程序。

本文为MPI学习中的一些 笔记，默认读者了解C/C++和Linux的基本知识。

<!—more—>

## MPI 安装

安装部分不做多的介绍。大部分的Linux发行版软件库中包含有MPI。自行安装即可。

## 点对点阻塞通信

所有的并行程序都会涉及到通讯问题。