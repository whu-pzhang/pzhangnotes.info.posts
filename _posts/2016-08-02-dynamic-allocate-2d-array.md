---
title: C语言中二维数组的动态分配
date: 2016-08-02 20:48:16
author: pzhang
categories: Programming
tags: [C, Linux]
---

在C语言中，数组时最常用到的。分为静态数组和动态数组。
静态数据即数组长度预先定义好，一旦给定大小就无法再改变长度，静态数组用完后会自动释放
内存。

动态数组的长度则是随程序的需要来指定的。其需要的内存由内存分配函数 `malloc` 或 `calloc`
从堆（heap）上分配，用完后需要程序员自己释放内存。

标准C语言中提供了一维数组动态分配和释放的函数，包含于头文件 `stdio.h` 中。

``` C
void *malloc(unsigned int size)
```
