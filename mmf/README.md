## 流程
首先建立一个trainer的容器，容器待加载完数据集、模型、损失函数等可以进行train、infer。
## 主要的模块
1. mmf/trainer/core/training_loop.py & evaluation_loop.py
模型训练迭代和评估的源码在这里
2. mmf/models/visual_bert.py
我们对visual_bert模型结构的修改都是在这个里面实现的
3. mmf/modules/losses.py
这里有该工程所有的损失函数的代码实现
4. mmf/utils/build.py
这里包括创建config、trainer、dataset、model等代码实现
5. mmf/configs/*
其中mmf/configs/defaults.yaml是最通用参数设置; 数据集的设置则在mmf/configs/datasets/hateful_memes/defaults.yaml中，包括train，val，test的设置；visual_bert的参数设置在mmf/configs/models/visual_bert/defaults.yaml中。

剩余的代码并没有做过什么修改。主要的修改集中在mmf/models/visual_bert.py和mmf/modules/losses.py这两个文件中
