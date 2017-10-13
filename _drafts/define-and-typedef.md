---
title: C 语言宏用法总结
author: pzhang
Date: 2017-10-13 16:12:12
categories: Programming
tags: C
---



## C语言宏用法

C语言中宏为预处理阶段的一种文本替换工具。分为对象类的宏和函数类的宏。

### 基础用法

对象类的宏仅仅为会被替换的代码片段，被称为对象类的宏也是因为其在代码中以数据对象的形式存在。

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

宏名之后带括号的宏被认为是宏函数。用法和普通函数一样，只不过其在预处理阶段就展开。
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


