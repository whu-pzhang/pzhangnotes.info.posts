---
title: C/C++ 中的 define 和 typedef
author: pzhang
date: 2017-10-13
lastMod: 2018-04-08
draft: false
categories:
  - Programming
tags:
  - c/c++
slug: define-and-typedef
---

c语言中，`#define` 和 `typedef` 均是用来定义别名的符号，但又有明显的不同。
`#define` 定义的宏只是简单的文本替换，`typedef` 则是类型别名。

<!--more-->

## 宏

C语言中宏为预处理阶段的一种文本替换工具。从使用上分为

- 对象类的宏

- 函数类的宏。

可以将任何有效的标识符定义为宏，你甚至可以将c语言关键字定义为宏，你能这么做的原因是因为c预处理器
没有关键字这个概念。利用这个特性你可以将`const`关键字对不支持的编译器隐藏起来。

### 基础用法

对象类的宏仅仅为会被替换为定义的代码片段，被称为对象类的宏也是因为其在代码中以数据对象的形式存在。

1. 标识符别名

```c
#define BUFFER_SIZE 1024
```

这是最常用的用法，预处理阶段，`BUFFER_SIZE`会被替换为`1024`。按照惯例，宏名用大写字母表示。

若宏体过长，可以加反斜杠换行

```c
#define NUMBERS 1,\
				2,\
				3
```

预处理阶段，`int x[] = { NUMBERS };`会被替换为`int x[] = { 1, 2, 3 };`

2. 宏函数

带括号的宏被认为是宏函数。用法和普通函数一样，只不过其在预处理阶段就展开。

```c
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
```

需要注意的是，宏名和括号之间不能有空格！

### 高级用法

#### 字符串化(Stringizing)

在写调试程序的宏函数时，我们希望可以将宏函数的参数转换为字符串嵌入字符常量中，这时我们可以使用符号`#`将其字符串化。

```c
#define WARN_IF(EXP) \
do { if (EXP) \
		 fprintf(stderr, "Warning: " #EXP "\n");} \
while (0)
```

这时`WARN_IF(x==0);`会被扩展成：

```c
do { if (x == 0)
    fprintf (stderr, "Warning: " "x == 0" "\n"); }
while (0);
```

上面宏函数体中的 `do {  } while (0)` 是在宏中有多个语句时用到的。为了将宏代码和其他片段分割开来。
譬如以下的程序：

```c
#define M() a(); b()

if (cond)
    M();
else
    c();

/* 预处理后 */

if (cond)
    a(); b();
else   /* <- else 没有对应的 if */
    c();
```

只用 `{}` 也不行：

```c
#define M() { a(); b(); }

if (cond)
    M();
else
    c();

/* 预处理后 */

if (cond)
    { a(); b(); };  /* 最后的分号表示 if 语句结束 */
else   /* <- else 没有对应的 if */
    c();
```

用 `do {} while(0)` 才可以：

```c
#define M() do { a(); b(); } while(0)

if (cond)
    M();
else
    c();

/* 预处理后 */

if (cond)
    do { a(); b(); } while(0);
else
    c();
```

#### 连接字符串(Concatenation)

当宏中出现`##`时，会对 token 进行连接：

```c
#define COMMAND(NAME)  { #NAME, NAME ## _command }

struct command
{
    char *name;
    void (*function) (void);
};

struct command commands[] =
{
    COMMAND (quit),
    COMMAND (help),
    ...
};
```

上述命令会被扩展为：

```c
struct command commands[] =
{
    { "quit", quit_command },
    { "help", help_command },
    ...
};
```

#### 变参数宏(Variadic Macros)

宏函数还可以像函数一样接受可变参数。语法和可变参数的函数一样：

```c
#define eprintf(...) fprintf(stderr, __VA_ARGS__)
```

这时调用宏的所有参数都会代替`__VA_ARGS__`被展开：

`eprintf("%s:%d: ", input_file, lineno)`会被展开为`fprintf(stderr, "%s:%d: ", input_file, lineno)`

以上宏定义不够直观，我们可以指定参数名：

```c
#define eprintf(fmt, ...) fprintf(stderr, fmt, __VA_ARGS__)
```

但是还有一个问题，调用上述宏定义时若省略可选参数，会报错。例如`eprintf("success!\n",);`会展开为`fprintf(stderr, "success!\n", );`，原因在于字符串末尾会多出来一个逗号。这时我们就需要用到前面提到的`##`了，将宏定义改写为：

```c
#define eprintf(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
```
当省略参数时，多余的逗号就会被删除。

## typedef

c语言中的 `typedef` 用来给数据类型取别名，目的是使代码易读和易理解。有简化声明的作用。

### 简化声明

定义了一个结构体如下：

```c
struct _Point {
  double x, y;
};
```

每次创建一个 `_Point` 类型的变量，得这么写：

```c
struct _Point a;
```

每次都要带上 `struct` 关键字增加击键数，这时可以利用 `typedef` 取别名简化变量声明。

在 `C++` 中定义结构体已经隐含了这层含义，因此可以直接使用 `_Point a;` 来声明变量。

```c
typedef struct _Point Point;
```

这样的话，定义新的变量可以写为

```c
Point a;
```

更易阅读和理解。

### 与数组和指针一起使用

先来看几个例子：

```c
typedef int iArr[6];

typedef struct Node Node;
struct Node {
  int data;
  Node *pNext;
};
typedef struct Node* pNode;

typedef int (*pfunc)(int a, int b);
```

这几个例子其实和c语言的复杂声明是一样的。去掉 `typedef` 关键字后就得到一个正常的
变量声明语句。

`typedef int iArr[6];` 变为 `int iArr[6];`，表示声明一个包含6个元素的`int`数组。
而加上 `typedef` 后就得到了一个新的类型名，`iArr` 不再是变量名，而是新的类型名 `iArr`，
用 `iArr` 可以去定义与原来的 `iArr` 变量相同类型的变量。

```c
iArr a;  /* a 为 包含6个元素的int数组 */
```

理解了这个后，剩下其他 `typedef` 就都好理解了。

```c
pNode pNew;  /* pNew 为指向 struct Node 类型的指针 */

int add(int a, int b) {
  return a + b;
}
pfunc func = add;  /* func 为一个函数指针，其指向的函数有两个int参数，返回int */
```

另外，在 `C++11` 标准中，引入了新的为类型取别名的关键字 `using`，可以用来代替 `typedef`，
而且更好理解：

```cpp
using iArr = int [6];
using pNode = Node *;
using pfunc = int (*) (int a, int b);
```

从可读性上 `using` 好于 `typedef` 。此外 `typedef` 和 `using` 不是完全等价的，using 可以用来
给模板取别买，`typdef` 则不行。

```cpp
template <typename T>
using Vec = MyVector<T, MyAlloc<T>>;

// usage
Vec<int> ivec;
```

用起来非常自然。若是使用 `typedef`，则是这样：

```cpp
template <typename T>
typedef MyVector<T, MyAlloc<T>> Vec;

// usage
Vec<int> vec;
```

编译的时候，会得到类似 `error: a typedef cannot be a template` 的错误信息。


总结起来，你只要弄懂了复杂声明，`typedef` 就很好理解！

此外，在 `C++11` 中推荐使用 `using` 代替 `typedef`。

## 参考

- [GNU Documents](http://gcc.gnu.org/onlinedocs/cpp/Macros.html)
