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

### MATLAB

### Madagscar

``` c
/* time-domain acoustic FD modeling */
#include <rsf.h>

int main(int argc, char* argv[])
{
    /* Laplacian coefficients */
    float       c11,c12,c21,c22,c0,tmp;
    bool        verb;           /* verbose flag */
    sf_file     Fw, Fv; /* I/O files */
    sf_axis     az,ax;    /* cube axes */
    int         it,iz,ix;        /* index variables */
    int         nt,nz,nx;
    int         sx, sz; // source position
    float       dt,dz,dx;
    float       fm;
    int         jt, ft;
    float       *wlt, **vv;     /* I/O arrays*/
    float       **u0, **u1, **ptr;/* tmp arrays */

    sf_init(argc, argv);
    if(! sf_getbool("verb",&verb)) verb=0;

    /* setup I/O files */
    Fv = sf_input ("in" );  // velocity model
    Fw = sf_output("out");  // wavefield snaps

    if (!sf_getint("nt", &nt)) sf_error("nt required");
    if (!sf_getfloat("dt", &dt)) sf_error("dt required");
    if (!sf_getint("ft", &ft)) ft=0;    // first recorded time
    if (!sf_getint("jt", &jt)) jt=1; // time interval
    if (!sf_getfloat("fm",&fm)) fm=20.0;

    /* Read/Write axes */
    az = sf_iaxa(Fv, 1);
    ax = sf_iaxa(Fv, 2);
    nz = sf_n(az);
    dz = sf_d(az);
    nx = sf_n(ax);
    dx = sf_d(ax);

    sf_oaxa(Fw, az, 1);
    sf_oaxa(Fw, ax, 2);
    sf_putint(Fw, "n3", (nt-ft)/jt);
    sf_putfloat(Fw, "d3", jt*dt);
    sf_putfloat(Fw, "o3", ft*dt);

    // source cooridnrate
    sx = nx/2;
    sz = nz/2;

    /* initialize 4-th order fd coefficients */
    tmp = 1.0/(dz*dz);
    c11 = 4.0*tmp/3.0;
    c12 = -tmp/12.0;
    tmp = 1.0/(dx*dx);
    c21 = 4.0*tmp/3.0;
    c22 = -tmp/12.0;
    c0 = -2.0*(c11+c12+c21+c22);

    wlt = sf_floatalloc(nt); // source wavelet
    vv = sf_floatalloc2(nz, nx);
    sf_floatread(vv[0], nz*nx, Fv);

    for (it=0; it<nt; it++) {
        tmp = SF_PI*fm*(it*dt-1.0/fm);
        tmp *= tmp;
        wlt[it] = (1.0-2.0*tmp)*expf(-tmp);
    }

    for(ix=0;ix<nx;ix++){
	    for(iz=0;iz<nz;iz++){
		    tmp=vv[ix][iz]*dt;
		    vv[ix][iz]=tmp*tmp;
	    }
	}

    /* allocate temporary arrays */
    u0 = sf_floatalloc2(nz, nx);
    u1 = sf_floatalloc2(nz, nx);

    for (iz=0; iz<nz; iz++) {
        for (ix=0; ix<nx; ix++) {
            u0[ix][iz]=0;
            u1[ix][iz]=0;
        }
    }

    /* MAIN LOOP */
    for (it=0; it<nt; it++) {
        if(verb) fprintf(stderr,"\b\b\b\b\b%d",it);

        if (it>=ft && (it-ft)%jt == 0) {
            sf_floatwrite(u0[0], nz*nx, Fw);
        }

        u1[sx][sz] += wlt[it];  // add source

        /* 4th order laplacian */
        for (ix=2; ix<nx-2; ix++) {
            for (iz=2; iz<nx-2; iz++) {
                tmp =
                    c0*u1[ix][iz] +
                    c11*(u1[ix][iz-1] + u1[ix][iz+1]) +
                    c12*(u1[ix][iz-2] + u1[ix][iz+2]) +
                    c21*(u1[ix-1][iz] + u1[ix+1][iz]) +
                    c22*(u1[ix-2][iz] + u1[ix+2][iz]);

                u0[ix][iz] = 2*u1[ix][iz] - u0[ix][iz] + vv[ix][iz]*tmp;
            }
        }

        ptr=u0;
        u0=u1;
        u1=ptr;

    }
    if(verb) fprintf(stderr,"\n");
    exit (0);
}
```
