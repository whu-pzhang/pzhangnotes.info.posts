---
title: C语言中二维数组的动态分配
date: 2015-08-02
lastMod: 2015-08-02
author: pzhang
categories:
  - Programming
tags:
  - C
  - Linux

slug: dynamic-allocate-2d-array
---

在C语言中，数组是最常用到的。分为静态数组和动态数组。
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

对于一维数组，直接使用这四个函数就可以了。但有的时候，我们需要动态的创建
二维数组或者三维数组。这时候就需要自行在这些函数的基础上，编写多维数组分配的函数。

## 分配内存不连续的空间

以二维整型数组 `int array[m][n]` 为例。

遵循从外到里，也就是从变化慢的维到变化快的维，逐层申请的原则。

最外层的指针就是数组名`array`，为一个指针的指针。

```c
// 给二维数组动态分配内存
int **p;
// 指针p指向数组array的第一维，有m个元素
p = (int **)malloc(m * sizeof(int *));
```

次层指针为 `array[]`, 是一个一维指针，直接对其分配内存就行了。

```c
for (int i=0; i<n; i++)
    p[i] = (int *)malloc(n * sizeof(int));
```

综合这两步：

```c
#include <stdlib.h>

int **alloc2int(size_t n1, size_t n2)
{
    int **p;
    if ((p = (int **)malloc(n1 * sizeof(int *))) == NULL) return NULL;
    for (int i=0; i<n1; i++) {
        if ((p[i] = (int *)malloc(n2 * sizeof(int))) == NULL) return NULL;
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

### 测试

我们可以测试看看，这样分配二维数组是否可行

```c
#include <stdio.h>
#define ROW 2
#define COL 3

int **alloc2int(size_t n1, size_t n2);
void free2int(int **p, size_t n1);

int main()
{
    int **m;

    m = alloc2int(ROW, COL);
    for (size_t i=0; i<ROW; i++)
        for (size_t j=0; j<COL; j++) {
            m[i][j] = i+j;
        }

    for (int i1=0; i1<ROW; i1++) {
        printf("m[%d] = %p\n", i1, m[i1]);
        for (int i2=0; i2<COL; i2++) {
            printf("&m[%d][%d] = %p; m[%d][%d] = %d\n", i1, i2, &(m[i1][i2]),
                i1, i2, m[i1][i2]);
        }
        printf("\n");
    }

    free2int(m, ROW);
    return 0;
}
```
编译链接：

    $ gcc -Wall -std=c99 test.c alloc.c
    $ ./a.out
    m[0] = 0x1433030
    &m[0][0] = 0x1433030; m[0][0] = 0
    &m[0][1] = 0x1433034; m[0][1] = 1
    &m[0][2] = 0x1433038; m[0][2] = 2

    m[1] = 0x1433050
    &m[1][0] = 0x1433050; m[1][0] = 1
    &m[1][1] = 0x1433054; m[1][1] = 2
    &m[1][2] = 0x1433058; m[1][2] = 3

可以看到，在二维数组的第一行和第二行中，内存是连续的，但是行与行之间内存就不连续了。
这样的话，我们就不能以一维数组指针的形式来对其进行取值，`*(*(a+i)+j)` 的值就不等于
`a[i][j]` 了。


## 分配内存连续的空间

我们知道，在静态的二维数组中，内存是连续的，也就是说可以将二维数组当作
特殊的一维数组来进行遍历。上面的方法明显做不到这一点。

为了使得内存连续。还需另外一种思路。

上述第一种方法内存不连续的原因在于，二维数组行指针的分配的在循环中依次分别进行的，
不能保证每个行指针之间相隔 `sizeof(int)*COL` 个字节。

那么，为了使得动态分配的二维数组内存也同静态二维数组一样是连续的，我们就需要将
每个行指针联系起来。

```c
#include <stdlib.h>

int **alloc2int(size_t n1, size_t n2)
// allocate a int matrix
{
    int **m;
    // allocate pointers to rows (m is actually a pointer to an array)
    // m 为一个行指针
    if ((m = (int **)malloc(n1*sizeof(int *))) == NULL) return NULL;
    // allocate rows and set pointers to them
    // m[0] 指向整个数组内存块的首地址
    if ((m[0] = (int *)malloc(n1*n2*sizeof(int))) == NULL) return NULL;
    for (size_t i1=1; i1<n1; i1++) m[i1]=m[i1-1]+n2;
    // return pointers to array of pointers to rows
    return m;
}

void free2int(int **m)
{
    free(m[0]);
    free(m);
}
```

### 测试

```c
#include <stdio.h>
#define ROW 2
#define COL 3

int **alloc2int(size_t n1, size_t n2);
void free2int(int **p);

int main()
{
    int **m;

    m = alloc2int(ROW, COL);
    for (int i=0; i<ROW; i++)
        for (int j=0; j<COL; j++) {
            m[i][j] = i+j;
        }

    int *start = *m; // *m = &m[0][0]
    int * const end = start + ROW*COL;
    for ( ; start != end; start++)
        printf("%p -> %d\n", start, *start);

    //printf("%p\n", m[0]+4);

    free2int(m);
    return 0;
}
```
编译链接：

    $ gcc -Wall -std=c99 test.c alloc.c
    $ ./a.out
    0x135d030 -> 0
    0x135d034 -> 1
    0x135d038 -> 2
    0x135d03c -> 1
    0x135d040 -> 2
    0x135d044 -> 3

可以看到，现在的二维数组内存是连续的了，我们用一个循环便可以遍历整个数组。

## 总结

对与二维动态数组的分配，我们要了解C语言中数组和指针的联系和区别，具体可以参考
[数组和指针](/array-and-pointers.html)

上述两种方法都可行，但是推荐用第二种方法！
