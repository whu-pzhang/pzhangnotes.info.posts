---
title: 时间域声波二维差分
date: 2016-05-27 11:12:54
modified: 2016-09-17
categories: 地震学基础
tags: 有限差分
---

在模拟地震波传播的方法中，有限差分(Finite-Difference)方法占有主导地位，并且在
地震勘探工业界以及结构模拟中扮演者越来越重要的角色。这是因为FD方法能够在较为复杂
的模型中 提供波场传播相对精确的“完全”解，与此同时，其计算效率也相对较高，并且比较
容易并行化。

本文介绍声波波动方程的有限差分格式。

<!--more-->

## 一维波动方程

一维标量波动方程：
$$ \frac{\partial^2 U}{\partial x^2} - \frac{1}{v^2} \frac{\partial^2 U}{\partial x^2} = f(t)$$



## 二维声波方程

二维声波方程表示如下：

$$ \Delta U - \frac{1}{v^2} \frac{\partial U^2}{\partial^2 t} = f(t) $$

变换一下写为：

$$  | \Delta U - f(t) | v^2 = \frac{\partial^2 U}{\partial t^2} $$

其中 $\Delta$ 为Laplacian算符， $f(t)$ 为震源函数，$v$为
速度，$U$ 为标量波场。

为了在计算机上实现，我们需要将该式子在时间和空间上离散化。具体实现就是利用Taylor级数
展开。本文只考虑时间二阶空间四阶的差分方程。


### 时间上的离散

为了将${\partial^2 U} / {\partial t^2}$离散化。首先将位移
$U_{i+1}$在时间步长上进行Taylor展开:

$$
U_{i+1} = U_i + \frac{\partial U}{\partial t} \Delta t +
\frac{1}{2} \frac{\partial^2 U}{\partial^2 t} (\Delta{t})^2 +
\frac{1}{6} \frac{\partial^3 U}{\partial^3 t} (\Delta{t})^3 + {higher\ order\ terms}
$$

其中$U_i$表示的不同时间的波场。
从中解出$\partial{U}/ \partial{t}$得到：

$$
\frac{\partial U}{\partial t} = \frac{1}{\Delta t} \big( U_{i+1} - U_i \big) -
\frac{1}{2} \frac{\partial^2 U}{\partial^2 t} \Delta{t} -
\frac{1}{6} \frac{\partial^3 U}{\partial^3 t} (\Delta{t})^2 - ...
$$

对其作近似得到：
$$
\frac{\partial U}{\partial t} \approx \frac{1}{\Delta{t}} \big( U_{i+1} - U_i \big)
$$
那么该近似的截断误差正比于$\Delta t$。

为了获得更好的近似，考虑$U_{i-1}$的Taylor展开式：

$$
U_{i-1} = U_i - \frac{\partial U}{\partial t} \Delta t +
\frac{1}{2} \frac{\partial^2 U}{\partial^2 t} (\Delta{t})^2 -
\frac{1}{6} \frac{\partial^3 U}{\partial^3 t} (\Delta{t})^3 + {higher\ order\ terms}
$$

同样解出$\partial{U}/ \partial{t}$得到：

$$
\frac{\partial U}{\partial t} = \frac{1}{\Delta t} \big( U_{i} - U_{i-1} \big) +
\frac{1}{2} \frac{\partial^2 U}{\partial^2 t} \Delta{t} -
\frac{1}{6} \frac{\partial^3 U}{\partial^3 t} (\Delta{t})^2 - ...
$$

将该式子与前面得到的$\partial{U}/ \partial{t}$联合起来得到：

$$
\frac{\partial U}{\partial t} = \frac{1}{2\Delta{t}} \big( U_{i+1} - U_{i-1} \big)
- \frac{1}{6} \frac{\partial^3 U}{\partial^3 t} (\Delta{t})^2 - ...
$$

近似得到：

$$\frac{\partial U}{\partial t} = \frac{1}{2\Delta{t}} \big( U_{i+1} - U_{i-1} \big)$$

可以看到现在的误差是正比于$(\Delta{t})^2$，比前面的近似误差小了。

同样的，我们将前面两个Taylor展开式相加可以得到位移对时间的二阶偏导近似表达式：

$$
    \frac{\partial^2{U}}{\partial{t^2}} \approx \frac{1}{(\Delta{t})^2} \big( U_{i+1}+U_{i-1}-2U_{i} \big)
$$

有了这个式子，我们就可以将开始的波动方程在时间上离散化，将其表示而二阶时间精度的差分形式：

$$ U_{i+1} = \big[ \Delta{U} - f(t) \big] v^2 (\Delta{t})^2 + 2U_{i} - U_{i-1} $$


### 空间上的离散

有了前面时间上离散作参考，空间上的离散其实是一样的，只不过我们会考虑更高阶次的近似。

同样的，将位移$U_{i+1}$和$U_{i-1}$对$x$方向作Taylor展开：

$$
U_{i+1} = U_i + \frac{\partial U}{\partial x} \Delta{x} +
\frac{1}{2} \frac{\partial^2 U}{\partial x^2} (\Delta{x})^2 +
\frac{1}{6} \frac{\partial^3 U}{\partial x^3} (\Delta{x})^3 +
\frac{1}{24} \frac{\partial^4 U}{\partial x^4} (\Delta{x})^4 +
\frac{1}{120} \frac{\partial^5 U}{\partial^5 x} (\Delta{x})^5 + {higher\ order\ terms}
$$

$$
U_{i-1} = U_i - \frac{\partial U}{\partial x} \Delta{x} +
\frac{1}{2} \frac{\partial^2 U}{\partial x^2} (\Delta{x})^2 -
\frac{1}{6} \frac{\partial^3 U}{\partial x^3} (\Delta{x})^3 +
\frac{1}{24} \frac{\partial^4 U}{\partial x^4} (\Delta{x})^4 -
\frac{1}{120} \frac{\partial^5 U}{\partial^5 x} (\Delta{x})^5 + {higher\ order\ terms} \\
$$

$$
U_{i+2} = U_{i} + \frac{\partial U}{\partial x} (2\Delta{x}) +
\frac{1}{2} \frac{\partial^2 U}{\partial x^2} (2\Delta{x})^2 +
\frac{1}{6} \frac{\partial^3 U}{\partial x^3} (2\Delta{x})^3 +
\frac{1}{24} \frac{\partial^4 U}{\partial x^4} (2\Delta{x})^4 +
\frac{1}{120} \frac{\partial^5 U}{\partial x^5} (2\Delta{x})^5 + {higher\ order\ terms} \\
$$

$$
U_{i-2} = U_{i} - \frac{\partial U}{\partial x} (2\Delta{x}) +
\frac{1}{2} \frac{\partial^2 U}{\partial x^2} (2\Delta{x})^2 -
\frac{1}{6} \frac{\partial^3 U}{\partial x^3} (2\Delta{x})^3 +
\frac{1}{24} \frac{\partial^4 U}{\partial x^4} (2\Delta{x})^4 -
\frac{1}{120} \frac{\partial^5 U}{\partial x^5} (2\Delta{x})^5 + {higher\ order\ terms}
$$

将上述四个式子两两相加再相减略去五阶以上的高阶项并消去四阶项可以得到：

$$
    \frac{\partial^2 U}{\partial x^2} = \frac{1}{(\Delta{x})^2} \big[  -\frac{1}{12} U_{i+2} + \frac{16}{12} U_{i+1} -
    \frac{30}{12} U_{i} +\frac{16}{12} U_{i-1} - \frac{1}{12} U_{i-2} \big]
$$

同理可以得到关于z的空间四阶差分格式如下：

$$
    \frac{\partial^2 U}{\partial z^2} = \frac{1}{(\Delta{z})^2} \big[  -\frac{1}{12} U_{i+2} + \frac{16}{12} U_{i+1} -
    \frac{30}{12} U_{i} +\frac{16}{12} U_{i-1} - \frac{1}{12} U_{i-2} \big]
$$

### 完整的差分格式

分别把位移对时间和空间x及z方向做了离散化之后，我们就可以得到Laplacian算符的差分形式：

$$
\begin{aligned}
\Delta{U} =& \frac{\partial^2 U}{\partial x^2} + \frac{\partial^2 U}{\partial z^2} \\\
          =& -\frac{1}{12 (\Delta{x})^2} \big( U_{m+2,n}^j + U_{m-2,n}^j \big) +
             \frac{16}{12 (\Delta{x})^2} \big( U_{m+1,n}^j + U_{m-1,n}^j \big) -
             \frac{30}{12 (\Delta{x})^2} \big( U_{m,n}^j \big) \\\
            &-\frac{1}{12 (\Delta{z})^2} \big( U_{m,n+2}^j + U_{m,n-2}^j \big) +
             \frac{16}{12 (\Delta{z}2)^2} \big( U_{m,n+1}^j + U_{m,n-1}^j \big) -
             \frac{30}{12 (\Delta{z})^2} \big( U_{m,n}^j \big)
\end{aligned}
$$


整理一下：

$$
\begin{aligned}
\Delta{U} =& -\frac{30}{12} * U_{m,n}^j * \big( \frac{1}{(\Delta{x})^2} + \frac{1}{(\Delta{z})^2} \big) \\\
           & +\frac{16}{12} * \big( U_{m+1,n}^j + U_{m-1,n}^j \big) * \frac{1}{(\Delta{x})^2} \\\
           & -\frac{1}{12} * \big( U_{m+2,n}^j + U_{m-2,n}^j \big) * \frac{1}{(\Delta{x})^2} \\\
           & +\frac{16}{12} * \big( U_{m,n+1}^j + U_{m,n-1}^j \big) * \frac{1}{(\Delta{z})^2} \\\
           & -\frac{1}{12} * \big( U_{m,n+2}^j + U_{m,n-2}^j \big) * \frac{1}{(\Delta{z})^2}
\end{aligned}
$$

其中$j$为时间上网格，$m$和$n$分别为x方向和z方向上的网格。
时间域二阶空间域四阶的波动
方程有限差分格式如下：

$$
    U_{m,n}^{j+1} =
$$

## 示例代码

在 Madagascar 下的声波空间4阶时间2阶差分程序如下：

``` c
/* time-domain acoustic FD modeling */
#include <rsf.h>
int main(int argc, char *argv[])
{
    // Laplacian coefficients
    float c0 = -30. / 12., c1 = +16. / 12., c2 = -1. / 12.;

    bool verb; // verbose flag
    sf_file Fw = NULL, Fv = NULL, Fo = NULL; // I/O files
    sf_axis at, az, ax; // cube axes
    int it, iz, ix; // index variables
    int nt, nz, nx;
    int jsnap, sx, sz;
    float dt, dz, dx, idx, idz;

    float *ww, **vv; // I/O arrays
    float **um, **uo, **up, **ud; // wavefield and laplacian arrays

    sf_init(argc, argv);
    if (!sf_getbool("verb", &verb)) verb = false;
    if (!sf_getint("jsnap", &jsnap)) jsnap = 10;
    // 震源位置
    if (!sf_getint("sx", &sx)) sx = 100;
    if (!sf_getint("sz", &sz)) sz = 100;

    // setup I/O files
    Fw = sf_input("in");
    Fo = sf_output("out");
    Fv = sf_input("vel");

    // Read/Write axes
    at = sf_iaxa(Fw, 1);
    nt = sf_n(at);
    dt = sf_d(at);
    az = sf_iaxa(Fv, 1);
    nz = sf_n(az);
    dz = sf_d(az);
    ax = sf_iaxa(Fv, 2);
    nx = sf_n(ax);
    dx = sf_d(ax);

    // 设置输出文件头段信息
    sf_oaxa(Fo, az, 1);
    sf_oaxa(Fo, ax, 2);
    sf_setn(at, nt / jsnap);
    sf_setd(at, jsnap * dt);
    sf_oaxa(Fo, at, 3);

    idz = 1 / (dz * dz);
    idx = 1 / (dx * dx);

    // read wavelet, velocity
    ww = sf_floatalloc(nt);
    sf_floatread(ww, nt, Fw);
    sf_fileclose(Fw);
    vv = sf_floatalloc2(nz, nx);
    sf_floatread(vv[0], nz * nx, Fv);

    // precompute v^2*dt^2
    for (ix = 0; ix < nx * nz; ++ix) {
        *(vv[0] + ix) *= *(vv[0] + ix) * dt * dt;
    }

    // allocate temporary arrays
    um = sf_floatalloc2(nz, nx);  // U@t-1
    uo = sf_floatalloc2(nz, nx);  // U@t
    up = sf_floatalloc2(nz, nx);  // U@t+1
    ud = sf_floatalloc2(nz, nx);  // laplace

    for (iz = 0; iz < nz; iz++) {
        for (ix = 0; ix < nx; ix++) {
            um[ix][iz] = 0;
            uo[ix][iz] = 0;
            up[ix][iz] = 0;
            ud[ix][iz] = 0;
        }
    }

    // MAIN LOOP
    if (verb) fprintf(stderr, "\n");
    for (it = 0; it < nt; ++it) {
        if (verb) sf_warning("%d;", it + 1);

        // 4th order laplacian
        for (iz = 2; iz < nz - 2; iz++) {
            for (ix = 2; ix < nx - 2; ix++) {
                ud[ix][iz] = c0 * uo[ix][iz] * (idx + idz) +
                    c1 * (uo[ix - 1][iz] + uo[ix + 1][iz]) * idx +
                    c2 * (uo[ix - 2][iz] + uo[ix + 2][iz]) * idx +
                    c1 * (uo[ix][iz - 1] + uo[ix][iz + 1]) * idz +
                    c2 * (uo[ix][iz - 2] + uo[ix][iz + 2]) * idz;
            }
        }

        // inject wavelet
        ud[sx][sz] -= ww[it];

        // scale by (v*dt)^2
        for (iz = 0; iz < nz; iz++) {
            for (ix = 0; ix < nx; ix++) {
                ud[ix][iz] *= vv[ix][iz];
            }
        }

        // time step
        for (iz = 0; iz < nz; iz++) {
            for (ix = 0; ix < nx; ix++) {
                up[ix][iz] = 2 * uo[ix][iz] - um[ix][iz] + ud[ix][iz];

                um[ix][iz] = uo[ix][iz];
                uo[ix][iz] = up[ix][iz];
            }
        }

        // write wavefield to output
        if (it % jsnap == 0) {
            sf_floatwrite(uo[0], nz * nx, Fo);
        }
    }
    if (verb) fprintf(stderr, "\n");
    free(ww);
    free(vv);
    free(*um);
    free(um);
    free(*uo);
    free(uo);
    free(*up);
    free(up);
    free(*ud);
    free(ud);

    exit(0);
}

```

相应的 `SConstruct` 如下：

``` python
from rsf.proj import *

Flow('vel', None,
     '''
    spike nsp=1 mag=2000
    n1=201 n2=201 d1=5 d2=5
    label1=Depth unit1=m label2=Lateral unit2=m
    ''')

Flow('wlt', None,
     '''
    spike n1=500 o1=0 d1=0.001 nsp=1 k1=100 mag=1 |
    ricker1 frequency=20 |
    window n1=500 |
    scale axis=123
    ''')

prog = Program('afd2d.c')
exe = str(prog[0])

Flow('snap', ['wlt', 'vel', exe],
     '''
    ${SOURCES[2].abspath} vel=${SOURCES[1]}
    verb=y jsnap=1 sx=100 sz=100
    ''')

Result('snap_0.3.rsf', 'snap',
       '''
    window n3=1 min3=0.3 |
    grey gainpanel=a color=j screenratio=1 title="wavefield at 0.3s"
    ''')

End()

```
波场快照如下：
![](/images/2016112900.jpg)
