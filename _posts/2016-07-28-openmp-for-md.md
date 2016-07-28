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

## PI值近似

对 $\pi$ 值的计算可以采用如下公式

$$ \frac{\pi}{4} = 1-\frac{1}{3}+\frac{1}{5}-\frac{1}{7}+... = \sum_{k=1}^{n} \frac{(-1)^k}{2k-1}$$

串行程序如下：
``` C
#include <stdio.h>
#include <time.h>

#define N 1000000000

int main()
{
    double factor;
    double pi_approx;

    clock_t t1 = clock();
    for (int i=0; i<N; i++) {
        factor = (i%2==0) ? 1.0 : -1.0;
        pi_approx += 4.0*factor/(2*i+1);
    }
    clock_t t2 = clock();
    double duration = (double)(t2-t1)/CLOCKS_PER_SEC;
    printf("It taken %.3f sec\n", duration);
    printf("The approxmate value of pi is %.8f\n", pi_approx);
    return 0;
}
```

编译运行：

    $ gcc -Wall -std=c99 -o pi pi
    $ ./pi
    It taken 3.530 sec
    The approxmate value of pi is 3.14159265

现在加上 `parallel for` 指令将其并行化:
