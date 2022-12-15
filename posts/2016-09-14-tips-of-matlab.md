---
title: MATLAB拾遗
author: pzhang
date: 2016-09-14
categories:
  - Programming
draft: true
tags:
  - MATLAB
---


记录一些MATLAB的函数用法。

<!--more-->
## inpolygon

官方文档：

> Points located inside or on edge of polygonal region

语法：
``` matlab
in = inpolygon(xq,yq,xv,yv)
[in, on] = inpolygon(xq,yq,xv,yv)
```

其作用就是判断点落在一个多边形内部还是正好在其边缘上。
其中，参数`xq`和`yq`表示点的横纵坐标，`xv`和`yv`则表示定义多边形的点坐标。
若点在该多边形内部，返回值`in`为1，反之为0；参数`on`为点是否在边缘上的逻辑值。

看一个例子：

``` matlab
% 定义一个多边形(五边形)
% 保证坐标第一个和最后一个表示的是同一点(即first=end)
xv = [1, 5, 9, 7, 4, 1];    % x坐标
yv = [5, 7, 6, 4, 3, 5];    % y坐标

% 产生100个0～10之间的随机数当作坐标值
rng default
xq = 10*rand(100,1);
yq = 10*rand(100,1);

% 判断这些随机点中哪些在多边形内
[in, on] = inpolygon(xq, yq, xv, yv);

% 绘图
figure
plot(xv, yv);
axis equal
hold on
plot(xq(in), yq(in), 'r+'); % 绘制落在多边形内的点
plot(xq(~in), yq(~in), 'bo');   % 落在多边形外的点
hold off
```

![](/images/2016091400.png)

上面的例子针对一个封闭曲面的多边形。对于铜钱形或者园环形同样也有解决方案。

下面来看一个官方的例子：

``` matlab
% 定义一个中间有矩形洞的矩形多边形.
% 顺时针指定外矩形的顶点，接着逆时针指定内矩形的顶点。
% 用NaN来分开内外矩形坐标
xv = [1 4 4 1 1 NaN 2 2 3 3 2];
yv = [1 1 4 4 1 NaN 2 3 3 2 2];

% 随机产生500个坐标
% rng default 将随机数产生器恢复至默认值，以便每次运行下面的
% 命令都能产生相同的随机数，使得结果可重复
rng default
xq = rand(500,1)*5;
yq = rand(500,1)*5;

in = inpolygon(xq,yq,xv,yv);

figure

plot(xv,yv,'LineWidth',2) % polygon
axis equal

hold on
plot(xq(in),yq(in),'r+') % points inside
plot(xq(~in),yq(~in),'bo') % points outside
hold off
```

![](/images/2016091401.png)
