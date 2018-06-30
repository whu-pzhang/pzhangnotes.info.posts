---
title: OpenCV数据类型
author: pzhang
date: 2018-06-30
lastMod:
categories:
  - 图像处理
tags:
  - OpenCV
  - c/c++

draft: true
slug: opencv-data-type
---

OpenCV数据类型可分为三类：

  - 基本数据类型（basic data type）

  由C++原始数据类型（`int`, `float`等）直接构成。
  这些基本类型包括简单的`vector`和`matrix`类以及表示几何的点、矩形、大小等。

  - 帮助对象（helper objects）

  表示更为抽象的概念，如垃圾回收类，用作切片的范围对象等。

  - 大数组类型（large array type）

  用来容纳数组或基本数据类型，例如`cv::Mat`

  
