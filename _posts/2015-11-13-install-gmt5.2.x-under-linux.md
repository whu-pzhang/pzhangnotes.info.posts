---
title: GMT 5.2.1 在 Linux 下的安装
date: 2015-11-13
author: pzhang
categories: GMT
tags: [安装, Linux]
---

转载于 Seisman [GMT 5.2.1 在 Linux 下的安装](https://seisman.info/install-gmt5.2.x-under-linux.html)

**注意：** 为 Ubuntu 和 CentOS 用户写了一个快速安装脚本，见
[gmt-easy-installer](https://github.com/gmt-china/gmt-easy-installer)。
有兴趣的可以试着安装，也欢迎测试各种不同的平台。

本文介绍如何在 Linux 下安装 GMT 5.2.x。

GMT 5.2.x 系列的安装与 GMT 5.1.x 系列的安装稍有不同，故而在前文基础上单独写一篇。

<!--more-->

说明：

1.  仅适用于 5.2.1
2.  所有命令均在一般用户下完成，需要 root 权限的命令都用 `sudo` 执行

## 下载

GMT 5.2.1 需要下载三个文件：

1.  GMT 源码： [gmt-5.2.1-src.tar.gz](ftp://ftp.soest.hawaii.edu/gmt/gmt-5.2.1-src.tar.gz)
2.  全球海岸线数据 GSHHG： [gshhg-gmt-2.3.5.tar.gz](http://gmt.soest.hawaii.edu/files/download?name=gshhg-gmt-2.3.5.tar.gz)
3.  全球数字图表 DCW： [dcw-gmt-1.1.2.tar.gz](ftp://ftp.soest.hawaii.edu/gmt/dcw-gmt-1.1.2.tar.gz)

GMT 5.2.1 的源码包中，没有自带官方的 PDF 文档，需要额外下载：

1.  [GMT_API.pdf](http://gmt.soest.hawaii.edu/doc/5.2.1/pdf/GMT_API.pdf)
2.  [GMT_Docs.pdf](http://gmt.soest.hawaii.edu/doc/5.2.1/pdf/GMT_Docs.pdf)
3.  [GMT_Manpages.pdf](http://gmt.soest.hawaii.edu/doc/5.2.1/pdf/GMT_Manpages.pdf)
4.  [GMT_Tutorial](http://gmt.soest.hawaii.edu/doc/5.2.1/pdf/GMT_Tutorial.pdf)

下载完成后，可以用 `md5sum` 检查压缩文件的 md5 值，以保证该文件是完整且未被篡改的:

    $ md5sum gmt-5.2.1-src.tar.gz gshhg-gmt-2.3.5.tar.gz dcw-gmt-1.1.2.tar.gz
    df96d3cc9a93bc3c049f1523ada57117  gmt-5.2.1-src.tar.gz
    12c6604d779ddd1727e96433da20e147  gshhg-gmt-2.3.5.tar.gz
    45c99d30026742dbc0b1644ea64f496d  dcw-gmt-1.1.2.tar.gz

## 依赖关系

### 基础依赖包

GMT 的编译需要 C 和 C++ 编译器、cmake 等开发工具。

对于 Ubuntu/Debian:

    sudo apt-get update
    sudo apt-get install gcc g++ cmake make libc6

对于 CentOS/RHEL/Fedora:

    sudo yum install gcc gcc-c++ cmake make glibc

GMT 的编译过程要求 cmake 的版本 `>=2.8.5` ，需要注意：

1.  安装 cmake 之后，可以通过 `cmake --version` 检查 cmake 版本；
2.  CentOS **6.5** 的官方源中 cmake 的版本为 2.6.4，版本过低，无法满足要求；
3.  CentOS **6.6** 的官方源中 cmake 的版本为 2.8.12，可以满足要求；
4.  使用 CentOS 6.5 的用户可以 `yum update` 将系统升级到 6.6，即可使用较高版本的 cmake；
5.  CentOS 6.5 用户若不愿意升级整个系统，则需要先安装 EPEL 源，再安装 EPEL 源中的 `cmake28` ，并且在接下来的编译过程中要将 `cmake` 命令改成 `cmake28`；
6.  CentOS 7 官方源中 cmake 版本为 2.8.11，可以直接安装使用；

### 软件依赖包

GMT5 的依赖包，相对来说要复杂很多：

-   看 PS 文件以及将 PS 转换成其他格式需要 `ghostscript`
-   网格文件需要 `netCDF` （\>=4.0，且需要支持 netCDF-4/HDF5）
-   GLib 库：提供了 C 语言下的树、hash、列表和字符串功能
-   Perl 兼容正则表达式库 [PCRE](http://www.pcre.org/)
-   地理空间数据抽象库 [GDAL](http://www.gdal.org/)
-   Fourier 变换库 [FFTW](http://www.fftw.org/) (\>=3.3)
-   线性代数库 LAPACK 和 BLAS
-   如果想要自行编译文档的话还需要 [Sphinx](http://sphinx-doc.org/) 以及 TeXLive

对于 Ubuntu/Debian:

    sudo apt-get update
    # 必须安装的包
    sudo apt-get install ghostscript libnetcdf-dev
    sudo apt-get install libglib2.0-dev
    # 推荐安装的包
    sudo apt-get install libgdal-dev python-gdal
    sudo apt-get install liblapack3
    # 可选的安装包
    sudo apt-get install libpcre3-dev libfftw3-dev

对于 RHEL/CentOS/Fedora:

    # 安装必须的包
    sudo yum install ghostscript netcdf-devel
    sudo yum install glib2-devel
    # 推荐安装的包
    sudo yum install gdal-devel gdal-python
    sudo yum install lapack64-devel lapack-devel
    # 可选的安装包
    sudo yum install pcre-devel fftw-devel

一些需要注意的地方:

1.  一定不要试图自己手动编译 netCDF，因为手动编译很难解决依赖问题，网上的大多数手动编译 netCDF 的教程中都关闭了 netCDF 对 HDF5 的支持，因而导致 GMT5 无法使用。如果在阅读本文之前曾经手动编译过，一定要将原来手动编译生成的文件删除干净。通常可以使用 `locate netcdf` 找到 `/usr/local` 目录下的与 netCDF 相关的文件，直接删除即可。
2.  CentOS 官方源中不带有 netCDF，需要先安装 EPEL 源
3.  pcre、fftw 和 gdal 不是必须要安装的，但是推荐安装。其中 gdal 在做数据格式转换时非常有用；
4.  其他发行版很久不用了，不清楚细节，读者可以在使用过程中补充。

## 安装 GMT

将之前下载的三个压缩文件以及四个 PDF 文档都放在同一个目录里，以下假定目录名为 `/home/seisman/Desktop/GMT`：

``` bash
# 当前目录名为 /home/seisman/Desktop/GMT
$ pwd
/home/seisman/Desktop/GMT

# 当前目录下包含了三个压缩文件和四个 PDF 文档
$ ls
dcw-gmt-1.1.2.tar.gz  gmt-5.2.1-src.tar.gz  gshhg-gmt-2.3.5.tar.gz
GMT_API.pdf  GMT_Docs.pdf  GMT_Manpages.pdf  GMT_Tutorial.pdf

# 解压三个压缩文件
$ tar -xvf gmt-5.2.1-src.tar.gz
$ tar -xvf gshhg-gmt-2.3.5.tar.gz
$ tar -xvf dcw-gmt-1.1.2.tar.gz

# 将 gshhg 和 dcw 数据复制到 gmt 的 share 目录下
$ mv gshhg-gmt-2.3.5 gmt-5.2.1/share/gshhg
$ mv dcw-gmt-1.1.2 gmt-5.2.1/share/dcw-gmt

# 将 PDF 复制到 doc_release 目录下
$ mv *.pdf gmt-5.2.1/doc_release/pdf

# 切换到 gmt 源码目录下
$ cd gmt-5.2.1

# 新建用户配置文件
$ gedit cmake/ConfigUser.cmake
```

向 `cmake/ConfigUser.cmake` 文件中加入如下语句:

    set (CMAKE_INSTALL_PREFIX "/opt/GMT-5.2.1")
    set (GMT_INSTALL_MODULE_LINKS FALSE)
    set (COPY_GSHHG TRUE)
    set (COPY_DCW TRUE)
    set (GMT_USE_THREADS TRUE)

-   `CMAKE_INSTALL_PREFIX` 设置 GMT 的安装路径，可以修改为其他路径
-   `GMT_INSTALL_MODULE_LINKS` 为 FALSE，表明不在 GMT 的 bin 目录下建立命令的软链接，也可设置为 TRUE
-   `COPY_GSHHG` 为 TRUE 会将 GSHHG 数据复制到 `GMT/share/coast` 下
-   `COPY_DCW` 为 TRUE 会将 DCW 数据复制到 `GMT/share/dcw` 下
-   `GMT_USE_THREADS` 表示是否开启某些模块的并行功能（CentOS 6.x 下，需设置为 `FALSE`）

以下几点说明，仅供高阶用户阅读：

1.  GMT 提供了用户配置的模板文件 `cmake/ConfigUserTemplate.cmake`，其中包含了众多可配置的变量以及大量的注释说明。可以直接将该文件名复制为 `cmake/ConfigUser.cmake`，然后在模板基础上做修改，以自定义 GMT 的安装。仅供高阶用户使用
2.  `CMAKE_INSTALL_MODULE_LINKS` 的作用是在 GMT 的 bin 目录下建立命令的软链接，以兼容 GMT4 语法，建议设置为 FALSE
3.  配置文件中 `GSHHG_ROOT` 和 `DCW_ROOT` 可以用于指定数据所在路径。此处已将数据放在 GMT 的 share 目录下，使得在配置过程中 GMT 可以自动找到，因而不需要设置这两个变量
4.  若系统中存在多个 GMT 的版本，按照上面的做法会存在多个 GSHHG 和 DCW 数据的副本，造成数据冗余。此时，可以将 gshhg 和 dcw 数据放在专门的目录中，比如 `/home/seisman/Datas/` 目录下。然后有两种解决办法：
    1.  完全按照上面的做法，在安装完成后，删除 `/opt/GMT-5.2.1/share` 目录下的 `coast` 和 `dcw` 两个目录，并建立两个指向数据的真实数据的软链接:

            $ cd /opt/GMT-5.2.1/share
            $ sudo rm -r coast/
            $ sudo rm -r dcw/
            $ sudo ln -s /home/seisman/Datas/gshhg-gmt-2.3.5 gshhg
            $ sudo ln -s /home/seisman/Datas/dcw-gmt-1.1.2 dcw

    2.  设置配置文件如下:

            set (CMAKE_INSTALL_PREFIX"/opt/GMT-5.2.1")
            set (GMT_INSTALL_MODULE_LINKS FALSE)
            set (GSHHG_ROOT"/home/seisman/Datas/gshhg-gmt-2.3.5")
            set (COPY_GSHHG FALSE)
            set (DCW_ROOT"/home/seisman/Datas/dcw-gmt-1.1.1")
            set (COPY_DCW FALSE)
            set (GMT_USE_THREADS TRUE)

继续执行如下命令以检查 GMT 的依赖关系:

    $ mkdir build
    $ cd build/
    $ cmake ..

`cmake ..` 会检查 GMT 对软件的依赖关系，我的检查结果如下:

    *  Options:
    *  Found GSHHG database       : /home/seisman/Desktop/GMT/gmt-5.2.1/share/gshhg (2.3.5)
    *  Found DCW-GMT database     : /home/seisman/Desktop/GMT/gmt-5.2.1/share/dcw-gmt
    *  NetCDF library             : /usr/lib64/libnetcdf.so
    *  NetCDF include dir         : /usr/include
    *  GDAL library               : /usr/lib64/libgdal.so
    *  GDAL include dir           : /usr/include/gdal
    *  FFTW library               : /usr/lib64/libfftw3f.so
    *  FFTW include dir           : /usr/include
    *  Accelerate Framework       :
    *  Regex support              : PCRE (/usr/lib64/libpcre.so)
    *  ZLIB library               : /usr/lib64/libz.so
    *  ZLIB include dir           : /usr/include
    *  LAPACK library             : yes
    *  License restriction        : no
    *  Triangulation method       : Shewchuk
    *  OpenMP support             : enabled
    *  GLIB GTHREAD support       : enabled
    *  PTHREAD support            : enabled
    *  Build mode                 : shared
    *  Build GMT core             : always [libgmt.so]
    *  Build PSL library          : always [libpostscriptlight.so]
    *  Build GMT supplements      : yes [supplements.so]
    *  Build GMT Developer        : yes
    *  Build proto supplements    : none
    *
    *  Locations:
    *  Installing GMT in          : /opt/GMT-5.2.1
    *  GMT_DATADIR                : /opt/GMT-5.2.1/share
    *  GMT_DOCDIR                 : /opt/GMT-5.2.1/share/doc
    *  GMT_MANDIR                 : /opt/GMT-5.2.1/share/man
    -- Configuring done
    -- Generating done

正常情况下的检查结果应该与上面给出的类似。若出现问题，则需要检查之前的步骤是否有误，检查完毕后重新执行 `cmake ..` ，直到出现类似的检查结果。检查完毕后，开始编译和安装:

    $ make -j
    $ sudo make -j install

## 自行编译文档

如果系统中安装了 sphinx 和 LaTeX，则可以自行编译文档。一般情况下，不建议自行编译文档，官方提供的文档已经足够:

    $ make -j docs_man          # 生成 man 文档
    $ make -j docs_html         # 生成 HTML 文档
    $ make -j docs_pdf          # 生成 PDF 文档
    $ make -j docs_pdf_shrink   # 生成更小的 PDF 文档
    $ sudo -j make install

## 修改环境变量

修改环境变量并使其生效：

``` bash
$ echo 'export GMT5HOME=/opt/GMT-5.2.1' >> ~/.bashrc
$ echo 'export PATH=${GMT5HOME}/bin:$PATH'>> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GMT5HOME}/lib64'>> ~/.bashrc
$ exec $SHELL -l
```

说明

-   第一个命令向 `~/.bashrc` 中添加环境变量 `GMT5HOME`
-   第二个命令修改 `~/.bashrc` ，将 GMT5 的 bin 目录加入到 `PATH` 中
-   第三个命令将 GMT5 的 lib 目录加入到动态链接库路径中，若为 32 位系统，则为 `lib` ；64 位系统则为 `lib64` ；
-   第四个命令是重新载入 bash，相当于 `source ~/.bashrc`
-   某些发行版下可能需要写入到 `~/.bash_profile` 而不是 `~/.bashrc`
-   某些发行版下可能需要退出再重新登陆，或关机重启

## 测试是否安装成功

在终端键入 `gmt` ，若出现如下输出，则安装成功:

    $ gmt --version
    5.2.1

Ubuntu 14.04/15.04 以及部分 Debian 用户，可能会出现如下信息:

    $ gmt
    Sub-commands for gmt:
    install    install more modules
    ERROR: Please specify valid params for 'gmt'.

出现该错误的原因是这几个发行版中的 `libgenome-perl` 包中提供了同名的命令 `/usr/bin/gmt` ，把该软件包卸载即可。

## 参考来源

1.  <http://gmtrac.soest.hawaii.edu/projects/gmt/wiki/BuildingGMT>
2.  [GMT4.5.14 在 Linux 下的安装](/install-gmt4-under-linux.html)
3.  [GMT5.1.2 在 Linux 下的安装](/install-gmt5-under-linux.html)

## 修订历史

-   2015-11-13：根据 5.1.2 的安装步骤更新至 5.2.1；
-   2015-12-23： `GMT_USE_THREADS` 功能需要安装 glib2 库文件；
