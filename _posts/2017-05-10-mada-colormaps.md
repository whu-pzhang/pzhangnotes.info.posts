---
title: Madagascar之colormap
tags:
  - madagascar
  - colormap
author: pzhang
categories: Exploration Seismology
date: 2017-05-10 00:00:00
---


平常作图时都会遇到如何选择colormap的问题。
而Madagascar文档也没有对各种colormap作系统的说明。

本文梳理了Madagascar中现有的各种colormap。

本文 `SConstruct` 脚本位于 [Github](https://github.com/whu-pzhang/Mada/tree/master/colormap).

<!--more-->

Madagascar中的配色方案在`sfgrey`和`sfgrey3`中由参数`color=`控制。
默认为灰阶的，参数为`color=i`

## 标准配色

Madagascar 中自带的配色方案如下表所示：

| 配色            | 参数      |
|:----------------|:----------|
| Rainbow         | `color=a` |
| Bone            | `color=b` |
| Cool            | `color=c` |
| Blue-white-red  | `color=e` |
| Flag            | `color=f` |
| Black-white-red | `color=g` |
| Hot             | `color=h` |
| Greyscale       | `color=i` |
| Jet             | `color=j` |
| Linear          | `color=l` |
| Pink            | `color=p` |
| Traffic         | `color=t` |


效果图如下：

![Different colorschemes](http://opq72e2wz.bkt.clouddn.com/mona.jpg)


将小写改为大写就可以获得翻转的配色方案。

而配色方案中加入`C`可以高亮显示被裁剪的值。

## 新加入的配色

除了上述基础配色，Madagascar中还有一些后续加入的配色方案。

### Light Bartlein

这种橙色，白色和紫色主导的配色提出在以下文章中：

Light, A. and P.J. Bartlein (2004) [The end of the rainbow? Color schemes for improved data graphics](http://geog.uoregon.edu/datagraphics/EOS/). EOS Transactions of the American Geophysical Union 85(40):385  

[Matteo Niccoli](https://mycarta.wordpress.com/2012/03/15/a-good-divergent-color-palette-for-matlab/)推荐其作为红-白-蓝（Madagascar中`color=g`的配色）地震数据显示配色的替代品。

该配色在Madagascar中的参数为`color=lb`。

![Light Bartlein](http://opq72e2wz.bkt.clouddn.com/lb.jpg)


### Cube-helix

这种配色被设计成根据其感知亮度单调增加，用黑白打印机打印时可以产生了很好的灰度。


该种配色在Madagascar中的参数为`color=x`。

![Cube-helix](http://opq72e2wz.bkt.clouddn.com/x.jpg)


### 其他配色


![Other color schemes](http://opq72e2wz.bkt.clouddn.com/colorbars.jpg)


## 自定义配色

目前够用，先不折腾了！

只知道自定义配色方案要用csv格式！

可以参考`rsf/tutorials/colormaps`

## 参考

1. [Color schemes](http://ahay.org/blog/2005/03/28/color-schemes/)
2. [How to evaluate and compare color maps](http://wiki.seg.org/wiki/How_to_evaluate_and_compare_color_maps)

## 更新

1. 2017年5月 初稿
2. 2018年3月 添加 Github 地址。
