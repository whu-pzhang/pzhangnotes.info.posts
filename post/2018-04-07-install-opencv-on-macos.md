---
title: macOS 安装 OpenCV
date: 2018-04-07T16:01:06+08:00
lastmod: 2018-04-07T16:01:06+08:00
author: pzhang
draft: false
categories:
  - 安装
tags:
  - macOS
  - OpenCV

slug: install-opencv-on-macos
---

记录macOS上安装OpenCV。

<!--more-->

## 依赖

- XCode
- Cmake
- Python
- NumPy

XCode 出厂自带。其他的可以通过 [HomeBrew](https://brew.sh/) 安装。

安装 XCode Command Line Tools

```bash
$ sudo xcode-select --install
```

## 从源码安装

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

```bash
ippFile=ippicv_2017u3_mac_intel64_general_20170822.tgz
ippHash=$(md5 $ippFile | cut -d" " -f4)
ippDir=.cache/ippicv

mkdir -p $ippDir
mv $ippFile $ippDir/$ippHash-$ippFile
```

编译安装：

```bash
mkdir release && cd release
cmake -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_OPENCV_PYTHON3=True \
  -D BUILD_OPENCV_PYTHON2=False \
  ..

make
sudo make install
```

## 从Homebrew安装

```shell
brew update
brew install opencv3 --with-python3
```

会自动下载一系列依赖，并安装OpenCV3。

## 测试

`C++`测试：

```cpp
#include <opencv2/opencv.hpp>

int main(int argc, char const *argv[])
{
    cv::VideoCapture cap(0);
    cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);

    cv::Mat frame;
    for (;;) {
        cap >> frame;
        cv::imshow("Test", frame);
        if (cv::waitKey(17) > 0) break;
    }

    return 0;
}
```

编译运行：

```bash
$ clang++ test.cpp $(pkg-config --libs opencv) -o test.x
$ ./test.x
```

会打开摄像头，按任意键退出。

Python3测试：

```bash
$ python3 -c "import cv2; print(cv2.__version__)"
3.4.1
```

至此已经成功安装OpenCV，并绑定了Python3。
