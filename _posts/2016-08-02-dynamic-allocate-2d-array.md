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

标准C语言中提供了一维数组动态分配和释放的函数，包含于头文件 `stdlib.h` 中。

<!--more-->

```c
extern void *malloc(unsigned int num_bytes);
extern void *calloc(int num_elems, int elem_size);
extern void *realloc(void *mem_address, unsigned int newsize);
extern void free(void *p);
```

对于一维数组，直接使用这四个函数就可以了。但有的时候，我们需要动态的创建二维数组或者三维数组。
这时候就需要自行在这些函数的基础上，编写多维数组分配的函数。

## 动态数组构建过程

以二维整型数组 `int array[m][n]` 为例。

遵循从外到里，也就是从变化慢的维到变化快的维，逐层申请的原则。

最外层的指针就是数组名`array`，为一个二维指针。

```c
// 给二维数组动态分配内存
int **p;
// 指针p指向数组array的第一维，有m个元素
p = (int **)malloc(m * sizeof(int *));
```

次层指针为 `array[]`, 是一个一维指针，直接对其分配内存就行了。

```c
for (int i=0; i<n; i++)
    array[i] = (int *)malloc(n * sizeof(int));
```

综合这两步：

```c
int **alloc2int(size_t n1, size_t n2)
{
    int **p;
    if ((p = (int **)malloc(n1 * sizeof(int *))) == NULL) return NULL;
    for (int i=0; i<n1; i++) {
        p[i] = (int *)malloc(n2 * sizeof(int));
    }
    return p;
}

void free2int(int **p, size_t n1)
{
    for (int i=0; i<n1; i++) {
        free(p[i]);
    }
    free(p);
}
```
