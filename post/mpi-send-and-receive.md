---
title: MPI学习 - 点对点通信
author: pzhang
date: 2018-04-08
lastMod: 2018-04-08
categories:
  - Programming
tags:
  - MPI
  - 高性能计算

draft: true
slug: mpi-send-and-receive
---

发送和接收信息是MPI的基础，本文为MPI阻塞点对点通信学习笔记。

<!--more-->

## MPI消息传递

MPI的发送和接收操作按如下步骤进行：

1. 进程A需要给进程B发送信息
2. A将信息打包进buff，并指定接收对象B
3. 通过网络发送打包好的buff
4. 进程B接收信息

MPI发送和接收函数的原型为：

```c
int MPI_Send(
    const void *buf,
    int count,
    MPI_Datatype datatype,
    int dest,
    int tag,
    MPI_Comm communicator)
```

```c
int MPI_Recv(
    void *buf,
    int count,
    MPI_Datatype datatype,
    int source,
    int tag,
    MPI_Comm communicator,
    MPI_Status *status)
```

看起来参数很多，但是很容易理解。和邮件的收发是类似的

- `buf`, `count` 和 `datatype`是被信封打包起来的信息
- `dest`: 收件人地址(进程ID)
- `source`: 发件人地址
- `tag`: 邮戳，可以区分邮件的顺序
- `communicator`: 邮局名称，收发邮件的邮局。

`MPI_Status` 为MPI定义的一个用来保存执行结果的结构体：

```c
typedef struct _MPI_Status {
  int count;        // 字节数
  int cancelled;
  int MPI_SOURCE;   // 接收方进程ID
  int MPI_TAG;
  int MPI_ERROR;
} MPI_Status
```

而 `MPI_Datatype` 也是MPI定义的数据类型，和c语言基本类型是一一对应的。

<!-- ## MPI基本数据类型

常用的MPI数据类型和C语言基本类型的对应关系如下：

| MPI datatype             | C equivalent                   |
|:-------------------------|:-------------------------------|
| `MPI_SHORT`              | `short int`                    |
| `MPI_INT`                | `int`                          |
| `MPI_LONG`               | `long int`                     |
| `MPI_LONG_LONG`          | `long long int`                |
| `MPI_UNSIGNED_CHAR`      | `unsigned char`                |
| `MPI_UNSIGNED_SHORT`     | `unsigned short int`           |
| `MPI_UNSIGNED`           | `unsigned int`                 |
| `MPI_UNSIGNED_LONG`      | `unsigned long int`            |
| `MPI_UNSIGNED_LONG_LONG` | `unsigned long long int`       |
| `MPI_FLOAT`              | `float`                        |
| `MPI_DOUBLE`             | `double`                       |
| `MPI_LONG_DOUBLE`        | `long double`                  |
| `MPI_BYTE`               | `char`                   | --> |

点对点通信要求`send`和`recv`要一一配对。
