---
title: OpenMP 学习笔记
date: 2016-07-24 20:44:13
author: pzhang
categories: Programming
tags: [OpenMP, Linux]
---

> OpenMP 是一个针对共享内存并行编程的 API 。
> 程序员和科学家开发 OpenMP， 使其在一个更高的层次来开发并行程序，其明确地被设计成用来对已有的串行程序进行增量式并行化，这对于 MPI 是不可能的。

本文为学习OpenMP的笔记，属于用多少学多少，学到哪里记到哪里，
不求大而全，只求学过的部分都弄明白。

<!--more-->

## 预备知识

OpenMP 提供“基于指令”的共享内存 API。 在 C 和 C++ 程序中加入预处理指令可以
用来允许不是基本 C 语言规范部分的行为，而不支持 `pragma` 的编译器就会忽略这些
语句，这样就允许使用 `pragma` 的程序在不支持它们的平台上运行。也就是说，
仔细编写的 `OpenMP` 程序可以在任何有 C 编译器的系统上运行。

### 编译和运行 OpenMP 程序

现在看一个简单的程序， `OpenMP` 版的 Hello World。
```C
#include <stdio.h>
#include <omp.h>

int main(int argc, char *argv[])
{
#pragma omp parallel
{
    int thread_id = omp_get_thread_num();
    int thread_count= omp_get_num_threads();

    printf("Hello from thread %d of %d\n", thread_id, thread_count);
}

    return 0;
}
```

现在的各类编译器基本上都提供对 OpenMP 的支持。只不过需要包含的选项不同。

使用 gcc 编译运行：

    $ gcc -g -Wall -fopenmp -o omp_hello omp_hello.c
    $ ./omp_hello
    Hello from thread 0 of 4
    Hello from thread 1 of 4
    Hello from thread 3 of 4
    Hello from thread 2 of 4

默认情况下，`OpenMP` 会使用所有的线程，可以通过
 `OMP_NUM_THREADS` 变量来控制使用的线程数：

    $ export OMP_NUM_THREADS=2
    $ ./omp_hello
    Hello from thread 1 of 2
    Hello from thread 0 of 2

另外可以注意到，输出不是按照线程编号顺序出现的，这是因为线程都在竞争访问
标准输出，其输出顺序不可预测的。

### 程序结构

从上述的 Hello World 程序我们可以看到，在 C/C++ 中 `OpenMP` 程序总是以
`#pragam omp` 作为开始。

在 `pragma` 后第一条指令为 `parallel`，表明之后的结构化代码（structured block）
块应该被多个线程并行执行。
运行结构化代码块的线程数将由运行时系统来决定。除了前面提到的 `OMP_NUM_THREADS` 变量可以指定线程数外，还
可以在 `parallel` 指令中使用 `num_threads` 子句来指定线程数。

    #pragma omp parallel num_threads(threads_count)

当程序到达 `parallel` 之前，程序只使用一个线程，到达 `parallel` 后，
原来的线程继续执行，另外 `threads_count-1`
个线程被启动。在 `OpenMP` 中，执行并行块的线程集和（原始线程和新的线程）
称为线程组（team），原始线程称为主线程（master），额外的线程称为
从线程（slave）。每个线程组的线程都执行 `parallel` 后的代码块。

代码块执行完后，有一个隐式路障（barrier）。也就是说当所有的线程完成了代码块，从线程将终止，主线程将继续执行之后的代码

### 错误检查

前面的例子中，我们没有进行错误检查。这在实际中是存在这风险的。

如果编译器不支持 `OpenMP` ，它将忽略 `#pragma` 指令，但是试图包含 `omp.h` 头文件
及调用 `omp_get_thread_num` 和 `omp_get_num_threads` 函数将会出错。 为了处理这些
问题，可以使用预处理器宏 `_OPENMP`。

``` C
#ifdef _OPENMP
#include <omp.h>
#endif
```

在后续的例子，为了简便起见，均不进行这种检查。但是实际运用中最好加上保证代码的
强健。


### 变量作用域

`OpenMP` 中，变量的作用域是对在 `parallel` 中线程而言。能够被线程组中所有线程
访问的变量拥有共享作用域，而只能被单个线程访问的变量拥有私有作用域。

在Hello World程序中，被每个线程使用的变量 `thread_id` 在每个线程的栈（私有）中
分配，因此其拥有私有作用域。

总之，在 `parallel` 前面已经声明的的变量，拥有共享作用域。在块中声明的变量拥有私有作用域。


## 临界区

先来看一个对 1-10000 求累加和的例子：
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
    for (int i=1; i<=N; i++) {
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
    $ ./omp_sum 2
    sum = 37514281
    $ ./omp_sum 4
    sum = 3126250

可以看到，我们用一个线程时，结果是对的，但是多个线程运行时就不对了。
这是为什么呢？
来分析一下：
`sum`是一个全局变量，所有的线程都可以访问，这样在多个线程运行时，会争相
对`sum`值进行改写，这样就会出现错误结果。
要解决这个问题，我们需要保证程序运行时每次只有一个线程执行 `sum += i` 语句。
这就涉及到 **临界区** 的概念了。

并行编程肯定会用到同步互斥操作。

**临界区就是指 在同一时刻只允许一个线程执行的代码块。**

在 `OpenMP` 中，临界区表示如下：

``` C
#pragma omp critical [(name)]
{
    // 代码块
}
```

将上面的代码改为：
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
    for (int i=1; i<=N; i++) {
#pragma omp critical (sum)
        sum += i;
    }
    printf("sum = %d\n", sum);
    return 0;
}
```

再编译执行，用多个线程也不会出错：

    $ ./omp_sum 2
    sum = 50005000
    $ ./omp_sum 4
    sum = 50005000

但是使用临界区相当于强制各个线程顺序执行，这样就会导致效率降低，其运行可能
会比串行程序慢。 OpenMP 提供更好的方法来避免线程串行执行： 归约子句(reduction).

## 归约子句(reduction)

**归约就是将相同的归约操作符重复地作用到操作序列上得到一个结果（归约变量）。**

对上面的例子，求和就是一个归约，归约操作符为加法。

还是上面的例子，我们使用归约操作：



## parallel for子句

### PI值近似

对 $\pi$ 值的计算可以采用[$\pi$的莱布尼茨公式](https://zh.wikipedia.org/wiki/%CE%A0%E7%9A%84%E8%8E%B1%E5%B8%83%E5%B0%BC%E8%8C%A8%E5%85%AC%E5%BC%8F)

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

    $ gcc -Wall -std=c99 -o pi pi.c
    $ ./pi
    It taken 3.530 sec
    The approxmate value of pi is 3.14159265

现在加上 `parallel for` 指令将其并行化:

``` C
#include <stdio.h>
#include <time.h>
#include <omp.h>
#define N 1000000000
int main()
{
    double factor;
    double pi_approx = 0.0;

    clock_t t1 = clock();
#pragma omp parallel for
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
