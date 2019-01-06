---
title: cs231n作业3笔记
author: pzhang
date: 2018-12-20
lastMod: 2018-12-20
markup: mmark
mathjax: true
categories:
  - 计算机视觉
tags:
  - Python
  - NumPy
  - PyTorch
  - TensorFlow

draft: true
slug: cs231n-assignment3
---

## 简介

在作业2中手撸CNN后，cs231n讲述了RNN，LSTM，GRU以及语言模型、图像标注、检测、定位、分割、识别等多方面的内容。


## RNN captioning

常规DNN和CNN都只能单独的处理一个个的输入，每个输入之间是完全没有联系的，这对于类似图像分类的任务来说是没有问题的，但某些任务需要更好地处理序列的信息，即前面的输入和后面的输入是有联系的，例如：

1. 图像标注问题(one to many)： image -> sequence of words
2. 情感分类(many to one): sequence of words -> sentiment
3. 机器翻译(many to many): seq of words -> seq of words
4. 帧级别的视频分类(many to many)
5. ……

为了更好地处理序列的信息，RNN(Recurrent Neural Network)就诞生了。
RNN通常要预测一个和时间相关的量，其基本架构如下：

![](/images/rnn.jpg)

每个时间步都要根据前一个状态和当前输入计算一个新的状态。

$$
\boldsymbol{h}_t = f_W (\boldsymbol{h}_{t-1}, \boldsymbol{x}_t) \\
\downarrow \\
\boldsymbol{h}_t = tanh (\mathbf{W}_{hh}\boldsymbol{h}_{t-1} + \mathbf{W}_{xh} \boldsymbol{x}_t)
$$

向上的箭头为每个时间步的输出，是一个softmax层

$$
\boldsymbol{y}_t = Softmax (\mathbf{W}_{hy} * \boldsymbol{h}_t)
$$

根据上述公式，RNN单元的 `rnn_step_forward` 如下：

```python
prev_hWh = prev_h @ Wh                          # (1)
xWx = x @ Wx                                    # (2)
hsum = prev_hWh + xWx + b                       # (3)
next_h = np.tanh(hsum)                          # (4)
```

根据计算图反推，backward也很简单：

```python
dhsum = (1 - np.tanh(hsum)**2) * dnext_h      # (4)
dprev_hWh = dhsum                             # (3)
dxWx = dhsum                                  # (3)
db = np.sum(dhsum, axis=0)                    # (3)
dx = dxWx @ Wx.T                              # (2)
dWx = x.T @ dxWx                              # (2)
dWh = prev_h.T @ dprev_hWh                    # (1)
dprev_h = dprev_hWh @ Wh.T                    # (1)
```

接下来就是完整的RNN了，forward过程需要对RNN单元循环 `T` 次（`T`为序列长度），
在循环内部每次记得更新 $\boldsymbol{h}_t$ 即可。

backward过程和以前的神经网络不太一样，RNN从上游传过来的梯度不只一个，除了从右边传过来的梯度外，还有每个时间点（上面）传过来的梯度。首先需要逆序循环，然后每个循环内更新RNN单元需要的梯度值。此外，在RNN中，Wx, Wh, b这三个参数是共享的，对每个时间步对视一样的，因此它们的梯度是一个累加的值。

```python
dprev_ht = np.zeros((N, H))
for t in reversed(range(T)):
    dx[:, t, :], dprev_ht, dWxt, dWht, dbt = rnn_step_backward(
        dh[:, t, :] + dprev_ht, cache[t])
    dWx += dWxt
    dWh += dWht
    db += dbt
dh0 = dprev_ht
```

在用RNN对图像进行标注前，还需要进行单词嵌入(word embeding)操作，因为神经网络不能将单词作为输入，所以需要将单词映射为单词索引的形式。

现在就可以来搭建RNN的架构了。首先需要把图像经CNN提取的特征（作业中采用的是在ImageNet数据集上预训练的VGG16模型的fc7层提取得到的特征）通过全连接层转换为初始的隐藏状态（$\boldsymbol{h}_0$），接着将captions做词嵌入输入RNN单元中，再利用一个仿射变换将RNN单元的输出转换为单词字典索引，接着采用softmax损失计算loss。

```python
h0, h0cache = affine_forward(features, W_proj, b_proj)
weout, wecache = word_embedding_forward(captions_in, W_embed)

if self.cell_type == 'rnn':
    rnnout, rnncache = rnn_forward(weout, h0, Wx, Wh, b)

x, xcache = temporal_affine_forward(rnnout, W_vocab, b_vocab)
loss, dx = temporal_softmax_loss(x, captions_out, mask)
```

上面描述的是RNN训练过程，在测试时就不一样了。测试时没有真值标注作为输入，需要一个起始单词（`<START>`）作为开始，然后按序列依次更新隐藏状态，并预测输出下一个标注单词。

```python
prev_h, _ = affine_forward(features, W_proj, b_proj)
captions[:, 0] = self._start
x = self._start * np.ones((N, ), dtype=np.int32)
for t in range(1, max_length):
    embed, _ = word_embedding_forward(x, W_embed)

    if self.cell_type == 'rnn':
        next_h, _ = rnn_step_forward(embed, prev_h, Wx, Wh, b)

    prev_h = next_h
    out, _ = affine_forward(next_h, W_vocab, b_vocab)
    idx = np.argmax(out, axis=1)
    captions[:, t] = idx
```

## LSTM captioning

RNN应该能够记住许多时间步之前见过的信息，但实际中由于梯度消失（vanishing gradient problem）问题，随着层数的增加，网络将变得无法训练。
LSTM(Long Short Term Memory)和GRU都是为了解决这个问题而提出的。
LSTM单元结构如下：

![](./images/lstm.png)

forward过程计算公式如下：

$$
\begin{align}
\begin{pmatrix}
i \\
f \\
o \\
g
\end{pmatrix} & =
\begin{pmatrix}
\sigma \\
\sigma \\
\sigma \\
\tanh
\end{pmatrix} \,
\mathbf{W}
\begin{pmatrix}
h_{t-1} \\
x_t
\end{pmatrix} \\
\\
c_t & = f \odot c_{t-1} + i \odot g \\
h_t & = o \odot \tanh(c_t)
\end{align}
$$

注意到每个LSTM单元有3个输入：$c, h, x$。实际计算时4个gate的线性部分可以通过一次计算完成，然后将不同gate对应的值分开即可。

```python
H = prev_h.shape[1]
z = x @ Wx + prev_h @ Wh + b
igate = sigmoid(z[:, :H])
fgate = sigmoid(z[:, H:2 * H])
ogate = sigmoid(z[:, 2 * H:3 * H])
ggate = np.tanh(z[:, 3 * H:])

next_c = fgate * prev_c + igate * ggate
next_h = ogate * np.tanh(next_c)
```

backward时，相比RNN，反传过来的值有两个 **`dnext_h`, `dnext_c`**。需要求：
`dx`, `dWx`, `dWh`, `db`, `dprev_h` 和 `dprev_c` 的梯度。
弄清楚哪些变量之间有关联，然后利用链式法则即可：

```python
dogate = dnext_h * np.tanh(next_c)
dnext_c += dnext_h * ogate * (1.0 - np.tanh(next_c)**2)
dfgate = dnext_c * prev_c
dprev_c = dnext_c * fgate
digate = dnext_c * ggate
dggate = dnext_c * igate

dz = np.zeros((N, 4 * H))
dz[:, :H] = digate * igate * (1.0 - igate)
dz[:, H:2 * H] = dfgate * fgate * (1. - fgate)
dz[:, 2 * H:3 * H] = dogate * ogate * (1.0 - ogate)
dz[:, 3 * H:4 * H] = dggate * (1.0 - ggate**2)

dx = dz @ Wx.T
dWx = x.T @ dz
dprev_h = dz @ Wh.T
dWh = prev_h.T @ dz
db = np.sum(dz, axis=0)
```

在完整的LSTM中大体和RNN相同，注意多了一个变量 `c`，不论是forward和backward都记得在循环里更新即可。

## Network Visualization
