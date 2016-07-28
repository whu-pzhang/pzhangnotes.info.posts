---
title: OpenMP 之 Hello World
date: 2016-07-24 20:44:13
author: pzhang
categories: Programming
tags: [OpenMP, Linux]
---

OpenMP 是一个针对共享内存并行编程的 API 。程序员和科学家开发 OpenMP， 使其在一个更高的
层次来开发并行程序，其明确地被设计成用来对已有的串行程序进行增量式并行化，这对于 MPI 是不可能的。

OpenMP 提供“基于指令”的共享内存 API。 在C和C++程序中加入预处理指令可以用来允许不是基本
C语言规范部分的行为，而不支持 `pragma` 的编译器就会忽略这些语句，这样就允许
使用 `pragma` 的程序在不支持它们的平台上运行。也就是说，仔细编写的 `OpenMP` 程序可以在
任何有C编译器的系统上运行。

<!--more-->

## 第一个OpenMP程序

### Hello World

程序语言的惯例，先来看一个 `OpenMP` 版的 Hello World程序：

``` C
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void Hello(void);

int main(int argc, char *argv[])
{
    #pragma omp parallel
    Hello();

    return 0;
}

void Hello(void)
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();

    printf("Hello from thread %d of %d\n", my_rank, thread_count);
}
```

编译运行：

    $ gcc -g -Wall -fopenmp -o omp_hello omp_hello.c
    $ ./omp_hello
    Hello from thread 0 of 4
    Hello from thread 1 of 4
    Hello from thread 3 of 4
    Hello from thread 2 of 4

默认情况下，`OpenMP` 会使用所有的线程，可以设置 `OMP_NUM_THREADS` 变量来控制使用的
线程数：

    $ export OMP_NUM_THREADS=2
    $ ./omp_hello
    Hello from thread 1 of 2
    Hello from thread 0 of 2

另外可以注意到，输出不是按照线程编号顺序出现的，这是因为线程都在竞争访问标准输出，其
输出顺序不可预测的。

### 程序结构

从上述的 Hello World程序我们可以看到，`OpenMP` 程序的 `#pragma` 总是以 `#pragam omp` 开始。

在 `pragma` 后第一条指令为 `parallel`，表明之后的结构化代码（structured block）块应该被多个线程并行执行。
运行结构化代码块的线程数将由运行时系统来决定。除了前面提到的 `OMP_NUM_THREADS` 变量可以指定线程数外，还
可以在 `parallel` 指令中使用 `num_threads` 子句来指定线程数。

    #pragma omp parallel num_threads(threads_count)

当程序到达 `parallel` 之前，程序只使用一个线程，到达 `parallel` 后，原来的线程继续执行，另外 `threads_count-1`
个线程被启动。在 `OpenMP` 中，执行并行块的线程集和（原始线程和新的线程）称为线程组（team），原始线程称为主线程
（master），额外的线程称为从线程（slave）。每个线程组的线程都执行 `parallel` 后的代码块。

代码块执行完后，有一个隐式路障。也就是说当所有的线程完成了代码块，从线程将终止，主线程将继续执行之后的代码

### 错误检查

前面的例子中，我们没有进行错误检查。这在实际中是存在这风险的。

如果编译器不支持 `OpenMP` ，它将忽略 `#pragma` 指令，但是试图包含 `omp.h` 头文件及调用 `omp_get_thread_num`
和 `omp_get_num_threads` 函数将会出错。 为了处理这些问题，可以检查预处理器宏 `_OPENMP` 是否定义。

``` C
#ifdef _OPENMP
#include <omp.h>
#endif
```
但在后续的例子，我均不进行这种检查。


## 变量作用域

`OpenMP` 中，变量的作用域是对在 `parallel` 中线程而言。能够被线程组中所有线程访问的变量拥有共享作用域，而只能
被单个线程访问的变量拥有私有作用域。

在Hello World程序中，被每个线程使用的变量 `my_rank` 在每个线程的栈（私有）中分配，因此其拥有私有作用域。
总之，在 `parallel` 前面已经声明的的变量，拥有共享作用域。在块中声明的变量拥有私有作用域。
