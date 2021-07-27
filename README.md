[![](https://img.shields.io/badge/Python->=3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/pandas-0.23.0-brightgreen.svg)](https://pypi.python.org/pypi/pandas/0.23.0)
[![](https://img.shields.io/badge/transformers-4.8.3-brightgreen.svg)](https://pypi.org/project/transformers/)
[![](https://img.shields.io/badge/pytorchlignting-1.3.8-brightgreen.svg)](https://pypi.org/project/pytorch-lightning/)
[![](https://img.shields.io/badge/pytorch->=1.4-brightgreen.svg)](https://pytorch.org/get-started/locally/)<br>

## 前言

&emsp;本工程会记录一些nlp领域的有趣应用

&emsp;由于目前的深度学习工程已经相当的模块化，为了整洁及便利性，该工程会考虑使用pytorch- lighting来组织训练。

```shell
/
|-- configs                   # 配置文件，可选
|-- datasets                  # 数据集存放路径
|-- datautils                 # 数据集处理脚本
|-- model                     # 模型目录
|-- module                    # 模型相关组件
|-- scripts                   # 训练及预测等脚步
|-- trainer									  # pytorch-learning训练框架
|-- utils                     
```



### 1.文本分类

&emsp;文本分类作为一项非常基础的工作，看起来似乎没那么好玩，所以这里也只是尝试了使用Bert作为分类器来使用，数据集为[THUCNews](http://thuctc.thunlp.org/)处理后的[语料库](https://github.com/649453932/Chinese-Text-Classification-Pytorch)，分类准确率98%左右。

### 2.文本生成

&emsp;生成类任务通常都是非常有趣的一个方向，其中，我最关注的两个细方向就是[人机对话](https://onedreame.github.io/2020/08/01/%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E6%A8%A1%E5%9E%8B%E6%BC%AB%E6%B8%B8/)以及[文本序列生成](https://onedreame.github.io/2021/03/31/%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E6%96%B9%E6%B3%95%E6%A2%B3%E7%90%86/)，理论性的东西见博客，这里记录一下工程层面的效果，

#### 2.1 文本序列生成

&emsp;这个任务指的是一些模型自主创作的任务，比如小说生成，诗词生成等等。

&emsp;在[transformer](https://onedreame.github.io/2020/09/06/transformer/)家族出现以前，这类任务基本由以*RNN-based*的*seq2seq*类模型统治，不过这类模型有一些问题，就是生成的效果不佳，表现为语句不够通顺，多样性差，安全回复居多等等。此外，由于*RNN*的序列特性，其还存在诸如梯度弥散等现象，总之各种因素结合起来，导致*seq2seq*虽然用的最频繁，但是其效果并不尽如人意.不过在一些对语句通顺性要求没那么高(诗歌)或者生成文本较短的情况下，*seq2seq*还是可以满足需要的。

#### 2.2 人机对话

&emsp;对于人机对话来说，存在着单轮和多轮之分，*seq2seq*处理单轮对话是非常自然的，不过在多轮回话层面则没有那么自然，目前已有的研究在使用seq2seq处理多轮对话的时候多采用 [*hierarchical recurrent encoder-decoder*](https://onedreame.github.io/2020/08/01/%E5%A4%9A%E8%BD%AE%E5%AF%B9%E8%AF%9D%E6%A8%A1%E5%9E%8B%E6%BC%AB%E6%B8%B8/)也即是层次化的seq2seq来建模多轮回话。

![](https://pic1.zhimg.com/80/v2-9fa6e8edbd67e885af75b273984b6b48_1440w.jpg)

&emsp;这类模型通常分为多层的*encoder*，底层的*encoder*用来对句子内建模，而高层的*encoder*则负责对句子间建模，这类模型由于采用了超多的*RNN*，所以需要精细的调参才可以收敛，而且收敛后的效果也不算多令人满意。

&emsp;不过，随着*GPT*为代表的*transformer*类模型的横空出世，多轮对话出现了很大的转机，对话效果有了比较大的提升，本工程也包含了*gpt2*训练多轮会话的实现，目录为$scripts/run\_generation.py$ 

> transformer类模型虽然效果不错，但是所需算力非常庞大，不要轻易尝试train from scratch，这点谨记。

&emsp;这里给出一些对话样例：

```python
>>你叫什么啊？
我叫什么？
>>是的，问你话呢
不知道
>>呵呵呵
你不是人
>>你咋骂人？
骂人怎么了
```

&emsp;transformer-based模型以庞大的参数换来了良好的效果，不过其性能问题比较严重，所需资源太多，使用受限，后续可通过蒸馏的方式去把transformer-based模型的性能迁移到轻量级的模型上去，是个不错的方向。
