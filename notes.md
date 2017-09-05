
## matplotlib

1. 存储图片时，可加入`bbox_inches='tight'` 参数切除白边

## Madagascar

1. kiss: Keep It Simple, Stupid.

## Git

1. `HEAD`: 表示当前版本
2. `reflog`: 记录历史命令
3. `git checkout -- file`: 丢弃工作区的改动。其实就是用版本库的并本替换工作区的
版本，无论是修改还是删除，都可以一键还原。
4. `git reset HEAD file`: 将暂存区的修改回退到工作区
5. `git reset --hard `: 版本回退
6. `git tag`: 打标签
    - `git tag <name>`用于新建一个标签，默认为`HEAD`，也可以指定一个`commit id`
    - `git tag -a <tagname> -m "blablabla..."` 可以指定标签信息
    - `git tag -s <tagname> -m "blablabla..."` `可以用PGP签名标签
    - `git tag` 可以查看所有标签。
7. `git rm -r --cached dirname` : 删除远程文件夹
