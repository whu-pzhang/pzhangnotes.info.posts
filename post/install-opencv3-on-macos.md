---
title: macOS 安装 OpenCV3
author: pzhang
date:
tag:

---

## 从源码安装

### 依赖

cmake，python3，numpy

### 下载

从[GitHub]((https://github.com/opencv/opencv/releases)
)下载最新的源码。

### 安装

推荐安装 Integrated Performance Primitives(IPP)，但是自动下载很慢，这里手动去
github下载：
https://github.com/opencv/opencv_3rdparty/tree/ippicv/

目前最新版本为 `ippicv_2017u3_mac_intel64_general_20170822.tgz`

下载完成之后将其放入OpenCV源码目录，为了避免cmake配置环境时重复下载，需要将
下载好的文件名中添加md5前缀然后放入 `.cache/ippicv` 文件夹内。

``` Bash
ippFile=ippicv_2017u3_mac_intel64_general_20170822.tgz
ippHash=$(md5 $ippFile | cut -d" " -f4)
ippDir=.cache/ippicv

mkdir -p $ippDir
mv $ippFile $ippDir/$ippHash-$ippFile
```


``` Bash
mkdir release && cd release
cmake -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_OPENCV_PYTHON3=True \
  -D BUILD_OPENCV_PYTHON2=False \
  ..

make
sudo make install
```

## 从Homebrew安装

brew install python3

brew install opencv3

```Bash
echo "/usr/local/opt/opencv/lib/python3.6/site-packages" >> /usr/local/lib/python3.6/site-packages/opencv3.pth
```
