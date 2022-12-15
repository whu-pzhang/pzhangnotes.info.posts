---
title: 添加程序到Madagascar中
date: 2016-11-03
lastMode: 2016-11-03
author: pzhang
categories:
  - 地震勘探
tags:
  - Madagascar
  - c/c++

slug: add-new-program-to-mada
---


有时候自己利用Madagascar编写了一些数据处理的程序，每次调用都需要将源码放到需要用的目
录重新编译，然后再在SConstruct中调用,类似这样：

```python
prog = Program('Mcode.c')
exe = str(prog[0])
# proj = Project()
# prog = proj.Program('Mcode.c')
Flow('out', ['inp1', 'inp2', exe],
    '''
    ${SOURCES[2].abspath} inp2=${SOURCES[1]}
    ''')
```

<!--more-->

若是只在一个project中用到倒还好，但若是比较通用的程序，在不同的文件夹中都要这么来调
用的话未免太麻烦。而将我们的代码直接加入Mada中，就可以直接调用。不用将源码拷来拷去还
要编译了。当然了，加入Mada的程序一般是我们的最终版本，测试的程序还是就放在测试目录中
比较好。

本文介绍怎么样将自己的程序加入到Madagascar中。


## 怎么样添加程序

添加程序简单的说，就以下两步：

- 在 `$RSFROOT/user` 目录下创建属于你自己的文件夹，通常以自己的名字命名，作为程序源
码文件夹，例如我的 `pzhang`
- 将自己的源码放入该目录
- 创建`SConstruct`文件进行编译
- 在`$RSFROOT` 目录下执行 `scons install`


## 一些规范

1. 主程序文件名必须以`M`开始，例如：`Mprog.c`或者`Mprog.py`。而且文件开始
最好加上对程序进行简短描述的注释。
2. 函数命名用小写字母开始。
3. 头文件`myprog.h`是根据`myprog.c`自动生成的。这里的规范有：
    - 对要加入头文件中的block，在其下方紧接着添加`/*^*/` 注释
    - 对于函数，在函数定义下添加 `/*< function description >*/` 注释

## 怎样添加注释

描述程序的注释

``` c
/* Short description line
Comments here blablabla lorem ipsum dolores sit amet...

You can use several paragraphs for comments, no problem.*/

/* Copyright notice */
```

参数注释：

```c
if (!sf_getbool("su",&su)) su=false;
/* y if input is SU, n if input is SEGY */
```

## 参考
- [Adding programs](http://www.ahay.org/wiki/Adding_new_programs_to_Madagascar)
