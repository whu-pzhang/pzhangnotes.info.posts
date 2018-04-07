---
title: finite-difference
tags:
---



## 微分方程的差分近似

**Before:**
- 泰勒级数
- 多步法

**After:**
- 确定不同差分的系数
- 根据精度设计不同的差分近似




### 有限差分近似

若函数 $U$ 可导，则其对 $x$ 的导数可表示为：
$$
\frac{\partial U}{\partial x} = \lim_{\Delta \rightarrow 0} \frac{U(x+\Delta x) - U(x-\Delta x)}{2 \Delta x }
$$



