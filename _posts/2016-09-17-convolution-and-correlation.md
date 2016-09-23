---
title: 卷积和相关
date: 2016-09-17 09:11:31
author: pzhang
categories: 地震学基础
tags: [MATLAB]
---

## 前言

一直对与卷积都不是太明白。后来接触接收函数，也只是知道卷积在频域为乘积。

<!--more-->
## 卷积

卷积是数学上的一种重要运算。

卷积定义：
$$ h(t) = f \ast g = \int_{-\infty}^{\infty} {f(\tau)g(t-\tau) \,\mathrm{d}t} $$

怎么理解卷积，知乎上有比较好的回答：
[怎样通俗易懂地解释卷积？](https://www.zhihu.com/question/22298352)

一句话总结：卷积就是加权叠加！

### 离散卷积

对于离散形式的卷积为：
$$
\begin{aligned}
h[i] = f \ast g &= \sum_{j=0}^{N-1} {f[j]g[i-j]} \\
                &= \sum_{j=0}^{N-1} {f[i-j]g[j]}
\end{aligned}
$$

卷积之后的数据长度为M+N-1，M和N分别为做卷积的两个序列的长度。

卷积满足交换律、分配律和结合律。

### 卷积计算方法
离散卷积计算主要有三种方法：

- 直接计算(Direct Method)
- 快速傅里叶变换(FFT)
- 分段卷积(sectioned convolution)

#### 直接计算

对于长度分别为M和N的序列，若$f[n]$和$g[n]$都为实数，则需要$MN$次乘法；
若f和g为i更一般性的复数信号，则需要4MN次乘法。

当两个信号都很长的时候，卷积直接计算是非常耗时的。

#### 快速傅立叶变换

因此，计算卷积通常都是在频域进行的，这主要是利用了卷积定理。
在时域做卷积相当于两个信号的傅里叶变换在频域做乘积。

可以将时域信号转换为频域信号，进行乘积运算之后再将结果转换为时域信号，实现快速卷积。

但是有一个问题：FFT运算假设其所计算的信号为周期信号，
因此通过上述方法计算出的结果实际上是两个信号的循环卷积，而不是线性卷积。
为了用FFT计算线性卷积，需要对信号进行补零扩展，使得其长度长于线性卷积结果的长度。

#### 分段计算

### 多项式乘积

其实卷积还可以和多项式乘积联系起来。

现有如下两个多项式：

$$
\begin{aligned}
f(x) &= a_0 + a_1 x + a_2 x^2 + a_3 x^3 \\
g(x) &= b_0 + b_1 x + b_2 x^2
\end{aligned}
$$

设$h(x)$为上述多项式的乘积，那么有：
$$
\begin{aligned}
h(x) &= (a_0 b_0) + \\
    &= (a_1 b_0 + a_0 b_1) x + \\
    &= (a_2 b_0 + a_1 b_1 + a_0 b_2) x^2\\
    &= (a_3 b_1 + a_2 b_2) x^3 \\
    &= (a_3 b_1 + a_2 b_2) x^4 \\
    & = a_3 b_2 x^5
\end{aligned}
$$

可以看到，乘积多项式各项的系数即为多项式f和g的系数的卷积！我们可以将其写成矩形相乘的形式：

$$
\left[
\begin{matrix}
c_0 \\
c_1 \\
c_2 \\
c_3 \\
c_4
\end{matrix}
\right] =
\left[
\begin{matrix}
b_0 & 0   & 0   & 0 \\
b_1 & b_0 & 0   & 0 \\
b_2 & b_1 & b_0 & 0 \\
0   & b_2 & b_1 & b_0 \\
0   & 0   & b_2 & b_1 \\
0   & 0   & 0   & b_2
\end{matrix}
\right]
\,
\left[
\begin{matrix}
a_0 \\
a_1 \\
a_2 \\
a_3
\end{matrix}
\right]
OR
=
\left[
\begin{matrix}
a_0     &   0   & 0   \\
a_1     & a_0   & 0   \\
a_2     & a_1   & a_0 \\
a_3     & a_2   & a_1  \\
0       & a_3   & a_2  \\
0       & 0     & a_3   
\end{matrix}
\right]
\,
\left[
\begin{matrix}
b_0 \\
b_1 \\
b_2
\end{matrix}
\right]
$$



## 相关

相关(Correlation)是概率论和统计学中用来刻画变量之间线性关系的强度和方向的。

通常所说的相关系数为:皮尔逊积差系数(Pearson's product moment coefficient)，这种相关系数
只对两个变量的线性关系敏感。

### 皮尔逊积差系数

公式表示如下：
$$
\rho_{X,Y} = \frac {cov(X,Y)}{\sigma_X \sigma_Y} = \frac{E((X-\mu_X)(Y-\mu_Y))}{\sigma_X \sigma_Y}
$$
其中，$E$为期望，$cov$表示协方差，$\sigma_X$和$\sigma_Y$为标准差。

由$\mu_X = E(x), \mu_X^2 = E(X^2)-E^2(X)$，上式可以写为：
$$
\rho_{X,Y} = \frac {E(XY)-E(X)E(Y)} {\sqrt{E(X^2)-E^2(X)} \sqrt{(E(Y^2)-E^2(Y))}}
$$

可以看出，只有两个变量的标准差都不为零，相关系数才有定义。而且相关系数的范围为[-1,1]

### 互相关

互相关(Cross-correlation)函数定义：
$$ h(t) = f \otimes g = \int_{-\infty}^{\infty} {f(t)g(t+\tau) \,\mathrm{d}t} $$

### 自相关

自相关(Auto-correlation)

### MATLAB函数

在matlab中，计算自相关和互相关的函数为`xcorr`
```matlab
c = xcorr(x);   % 自相关
c = xcorr(x,y)  % 互相关
```

## 卷积和相关的区别

## 参考

- [卷积](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF)
- [Python科学计算-快速卷积](http://old.sebug.net/paper/books/scipydoc/frequency_process.html#id5)
- [相关](https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%85%B3)
