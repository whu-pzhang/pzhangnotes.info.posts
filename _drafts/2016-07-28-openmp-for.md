---
title: OpenMP 之 parallel for
author: pzhang
date: 2016-07-28 20:09:00
categories: Programming
tags: [OpenMP, Linux]
---

在上一篇，对 `OpenMP` 做了简单的介绍。

前面提到，在 OpenMP 中只需要对原始串行程序做较少的改动便能对其并行化。具体就是添加
一些以 `#pragma omp` 开头的语句。

其中最常用到的就是 `parallel for` 语句了。

<!--more-->

## 累加求和

现在我们要求1-10000的累加和，可以编写程序如下：
``` C
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000

int main(int argc, char *argv[])
{
    int sum=0;
    int nthreads = strtol(argv[1], NULL, 10);
#pragma omp parallel for num_threads(nthreads)
    for (int i=1; i<=N； i++) {
        sum += i;
    }
    printf("sum = %d\n", sum);
    return 0;
}
```

编译链接，先用一个线程运行：

    $ gcc -Wall -std=c99 -fopenmp -o omp_sum omp_sum.c
    $ ./omp_sum 1
    sum = 50005000
    $ ./omp_sum 4
    sum = 8659995
    $ ./omp_sum 4
    sum = 3126250

可以看到，我们用一个线程时，结果是对的，但是多个线程运行时就不对了。这是为什么呢？
来分析一下：`sum`是一个全局变量，所有的线程都可以访问，这样在多个线程运行时，会争相
对`sum`值改写，这样就会出现结果错误。也就是说，我们需要保证每次只有一个线程执行
`sum += i` 语句。这就涉及到 **临界区** 的概念。

## 临界区

并行编程肯定会用到同步互斥操作。

临界区就是指 在同一时刻只允许一个线程执行的代码块。

在 `OpenMP` 中，临界区表示如下：

``` C
#pragma omp critical [(name)]
{
    // 代码块
}
```
