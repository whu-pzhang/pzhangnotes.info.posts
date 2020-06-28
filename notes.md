---
author: pzhang
date: 2017-10-06
lastMod: 2018-04-07
toc: true

slug: notes
---

## matplotlib

1. 存储图片时，可加入`bbox_inches='tight'` 参数切除白边


## Git

1. `HEAD`: 表示当前版本

2. `reflog`: 记录历史命令

3. `git checkout -- file`: 丢弃工作区的改动。其实就是用版本库的并本替换工作区的版本，无论是修改还是删除，都可以一键还原。

4. `git reset HEAD file`: 将暂存区的修改回退到工作区

5. `git reset --hard `: 版本回退

6. `git tag`: 打标签
    - `git tag <name>`用于新建一个标签，默认为`HEAD`，也可以指定一个`commit id`
    - `git tag -a <tagname> -m "blablabla..."` 可以指定标签信息
    - `git tag -s <tagname> -m "blablabla..."` 可以用PGP签名标签
    - `git tag` 可以查看所有标签。

7. `git rm -r --cached dirname` : 删除远程文件夹

8. fork 他人的 repo，做了自己的修改，在不冲突的前提下保持与上游项目同步：

    ```bash
    git remote add upstream https://github.com/someone-repo.git  # 配置上游项目地址
    git fetch upstream  # 获取上游项目更新
    git merge upstream/master  # 合并到本地分支
    git push origin master  # 提交推送到自己的 repo
    ```

9. 更改远程分支地址

    ```bash
    git remote rm origin # 如不需要原来的地址，可删除
    git remote set-url origin new-url
    ```

10. 添加子模块

    ```bash
    # 将远程repo添加为当前repo下的子模块
    git submodule add -b master https://github.com/someone-repo.git dirname
    # 保持子模块和远程分支同步
    git submodule update --recursive --remote
    ```

## tmux

1. 给当前窗口改名:

    `C-b` then`·:rename-window <newname>` 或者 `tmux renane-window <newname>`

2. 窗口滚动

    `C-b` then `[`  然后就可以用方向键或者鼠标滚轮进行上下滚动了


## Madagascar

1. 编译程序时添加链接库更改程序后缀：

```python
from rsf.proj import *

proj = Project()
proj.Prepend(LIBS='rsfgee rsfpwd')
proj.Replace(PROGSUFFIX='.x')

prog = proj.Program('Mprog.c')
exe = str(prog[0])

Flow('out', 'in '+exe,
    '''
    ${SOURCES[1].abspath} verb=y blablabla...
    ''')
End()
```

## 其他

1. 取模和取余

整数 a 和 b
`MOD` 和 `REM` 的计算都分两步：

- `c = a / b`
- `r = a - (c * b)`

不同之处在第一步里面：

- 取模时，`c` 往负无穷取整
- 取余时，`c` 往靠近零的方向取整

2. arXiv加速

将 `https://arxiv.org`替换为 `http://xxx.itp.ac.cn`

3. Github克隆加速

将 `github.com` 替换为 `github.com/cnpmjs.org`

