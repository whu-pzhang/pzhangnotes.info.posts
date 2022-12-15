---
title: 遥感数据集
author: pzhang
date: 2021-10-10
categories:
  - 遥感
  - 深度学习
tags:
  - dataset

draft: true
slug: rs-datasets
---


## 场景分类

| Dataset                                                              | # Class | #Images per cat. | # Instances | Resolution(m) | Image size | Year |
| -------------------------------------------------------------------- | :-----: | :--------------: | :---------: | ------------- | ---------- | ---- |
| [UC-Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |   21    |       100        |    2100     | 0.3           | 256        | 2010 |
| [WHU-RS19](http://www.graphnetcloud.cn/1-22)                         |   19    |      50~61       |    1013     | up to 0.5     | 600        | 2012 |
| [AID](https://captain-whu.github.io/AID/)                            |   30    |     220~420      |   10,000    | 0.5 to 8      | 600        | 2017 |
| [RSD46-WHU](https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU)          |   46    |     500~3000     |   117,000   | 0.5 to 2      | 256        | 2017 |
| [BigEarthNet](https://bigearth.net/)                                 |   43    |   320~217,119    |   590,326   | 10,20,60      | 20;60;120  | 2019 |
| [MLRSN](https://data.mendeley.com/datasets/7j9bv9vwsx/3)             |   46    |    1500~3000     |   109,161   | 0.1 to 10     | 256        | 2020 |


相关文献：

- Yang Y ,  Newsam S . Bag-of-visual-words and spatial extensions for land-use classification[C]// Sigspatial International Conference on Advances in Geographic Information Systems. ACM, 2010:270.
- G.-S. Xia, W. Yang, J. Delon, Y. Gousseau, H. Sun, and H. Maı ˆtre, “Structural high-resolution satellite image indexing,” in Proc. ISPRS TC VII Symposium - 100 Years ISPRS, 2010, pp. 298–303.
- G.-S. Xia, J. Hu, F. Hu, B. Shi, X. Bai, Y. Zhong, L. Zhang, and X. Lu, “AID: A benchmark data set for performance evaluation of aerial scene classification,” IEEE Trans. Geosci. Remote Sens., vol. 55, no. 7, pp. 3965–3981, 2017.
- Z. Xiao, Y. Long, D. Li, C. Wei, G. Tang, and J. Liu, “High-resolution remote sensing image retrieval based on cnns from a dimensional perspective,” Remote Sens., vol. 9, no. 7, p. 725, 2017.
- G. Sumbul, M. Charfuelan, B. Demir, and V. Markl, “BigEarthNet: A large-scale benchmark archive for remote sensing image understanding,” in Proc. IEEE Int. Geosci. Remote Sens. Symp., 2019, pp. 59015904.
- X. Qi, P. Zhu, Y. Wang, L. Zhang, J. Peng, M. Wu, J. Chen, X. Zhao, N. Zang, and P. T. Mathiopoulos, “MLRSNet: A multi-label high spatial resolution remote sensing dataset for semantic scene understanding,” ISPRS J. Photogrammetry Remote Sens., vol. 169, pp. 337–350, 2020.
  




## 语义分割

| Dataset                                                                                               | # Class | #Images | Resolution(m) | # Channels      | Image size | Year |
| ----------------------------------------------------------------------------------------------------- | :-----: | :-----: | :-----------: | --------------- | ---------- | ---- |
| [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) |    6    |   33    |     0.09      | IR,R,G,DSM,nDSM | ~2500      | 2012 |
| [ISPRS Postdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)     |    6    |   38    |     0.05      | IR,R,G,DSM,nDSM | 6000       | 2012 |
| [Massachusetts Buildings](https://www.cs.toronto.edu/~vmnih/data/)                                    |    2    |   151   |       1       | RGB             | 1500       | 2013 |
| [Massachusetts Roads](https://www.cs.toronto.edu/~vmnih/data/)                                        |    2    |  1171   |       1       | RGB             | 1500       | 2013 |
| [Inria](https://project.inria.fr/aerialimagelabeling/)                                                |    2    |   360   |      0.3      | RGB             | 5000       | 2017 |
| [DSTL-SIFD](https://www.kaggle.com/competitions/dstl-satellite-imagery-feature-detection/data)        |   10    |   57    |   up to 0.3   | up to 16        | ~3350x3400 | 2017 |
| [DeepGlobe Land Cover](https://competitions.codalab.org/competitions/18468)                           |    7    |  1146   |      0.5      | RGB             | 2448       | 2018 |
| [95-Cloud](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset)                     |    1    |  43902  |      30       | NIR,RGB         | 384        | 2019 |
| [UAVid](https://uavid.nl/#download)                                                                   |    8    |   420   |       -       | RGB             | ~4000x2160 | 2020 |
| [GID](https://x-ytong.github.io/project/GID.html)                                                     |   15    |   150   |    0.8~10     | NIR,RGB         | 6800x7200  | 2020 |
| [LoveDA](https://github.com/Junjue-Wang/LoveDA)                                                       |    7    |  5987   |      0.3      | RGB             | 1024       | 2021 |
| [FloodNet](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021)                             |   10    |  2343   |       -       | RGB             |            | 2021 |


Note:

- The DSTL-SIFD is short for the challenge of Dstl Satellite Imagery Feature Detection


相关文献：

- 

## 目标检测

| Dataset                                                                      | Annot. | # Class | # Instances | # Images | Resolution(m) | Image width | Year |
| ---------------------------------------------------------------------------- | :----: | :-----: | :---------: | :------: | :-----------: | ----------- | ---- |
| NWPU-VHR10                                                                   |  HBB   |   10    |    3651     |   800    |   0.08 to 2   | ~1000       | 2014 |
| UCAS-AOD                                                                     |  OBB   |    2    |    14596    |   1510   |       -       | ~1000       | 2015 |
| [HRSC2016](http://www.escience.cn/people/liuzikun/DataSet.html)              |  OBB   |   26    |    2976     |   1061   |       -       | ~1100       | 2016 |
| DIOR                                                                         |  HBB   |   20    |   192472    |  23463   |   0.5 to 30   | 800         | 2019 |
| [DOTA-v1.0](https://captain-whu.github.io/DOTA/)                             |  OBB   |   15    |   188282    |   2806   |   up to 0.3   | 800~13000   | 2018 |
| [DOTA-v1.5](https://captain-whu.github.io/DOTA/)                             |  OBB   |   16    |   402,089   |   2806   |   up to 0.3   | 800~13000   | 2019 |
| [DOTA-v2.0](https://captain-whu.github.io/DOTA/)                             |  OBB   |   18    |  1,793,658  |  11,268  |   up to 0.3   | 800~20000   | 2020 |
| [NWPU VHR-10](http://jiong.tea.ac.cn/people/JunweiHan/NWPUVHR10dataset.html) |  HBB   |   10    |    3651     |   800    |               | ~1000       |
| [VEDAI](https://downloads.greyc.fr/vedai/)                                   |  OBB   |    3    |    2950     |   1268   |               | 512,1024    |
| [ODAI](https://captain-whu.github.io/ODAI/)                                  |        |   15    |             |          |               |             |



## 实例分割

### iSAID

## 变化检测



## 视频跟踪

### SatVideoDT 2022

基于卫星视频数据的移动目标检测与跟踪数据集

https://satvideodt.github.io/

ICPR 2022: The 1st Challenge on Moving Object Detection and Tracking in Satellite Videos

分为三个赛道任务：

1. 卫星视频中的运动目标检测
2. 卫星视频中的单目标跟踪
3. 卫星视频中的多目标跟踪


| Dataset                                     | Annot. | # Class | # Video | # Frames | Resolution(m) | Image width | Year |
| ------------------------------------------- | :----: | :-----: | :-----: | :------: | :-----------: | ----------- | ---- |
| [SatVideoDT](https://satvideodt.github.io/) |  HBB   |   10    |   100   |  32,825  |     1.13      | 12000x5000  | 2022 |



