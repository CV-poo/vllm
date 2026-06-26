# 4\-\-int8 kvcache性能优化项目

**int8 kvcache性能优化**

# **一、项目背景**

目前的量化对象有权重、激活值和kvcache量化。俗称w8a8c8，weight、activation、cache。目前w8a8已经有成熟的方案了。cache8，现在在fp8场景下也有对应的实践了。

但是现在很多国产芯片公司没有原生的fp8算力，所以他们需要int8的量化实现。这就是项目的需求背景。这套方案涉及算子、kvcache和PD分离框架，涉及的面比较广。

# **二、项目内容**

这个方案的核心是让decode阶段的pageattention/mla这类attention算子的kvcache为int8数据类型。这样能大幅降低attention的计算时长，包括访存优化，以及计算优化。所以核心性能收益来源是对应的int8 kvcache的算子。怎么在保证有性能收益的同时，还能精度不下降。这里最重要的是精度，精度如果不准，性能再高也没用。所以核心还是在int8 kvcache算子。

量化是怎么做量化？选定量化参数，然后用量化公式将所有的数值都量化到\-128到127。怎么选量化参数？第三四讲合辑的文档里有提到，kvcache的shape为\[block，block\_size，2，head，head\_dim\]。一共就这么几个维度，需要根据实际的kvcache分布，选择在哪个维度做量化，比如在整个kvcache选择最大值作为量化参数，这就是perTenor量化。也可以在token维度，每个token一个量化参数，这就是perToken量化。也可以在head维度，每个head一个量化参数，这就是perHead量化，还可以在head \* head\_dim上，每个head \* head\_dim（一个token的特征向量）一个量化参数，这也叫做perChannel。不同量化维度对应的量化参数的shape不一样，而且就有对应不同的算子实现。

那究竟怎么选？先看kvcache的分布，统计出分布图，哪个分布集中就在哪个维度上做量化。选好了，就可以开始写算子，此处可以先用pytorch拼出一个等效的实现，算子最后再写，先把整个链路跑通。

假设已经有了int8 kvcache算子（pytorch实现），以qwen模型为例，如何在框架中支持这个算子呢？

首先就是先跑动态量化确定精度没问题。在调用pageattention的地方，此时输入的q和kvcache都是fp16/bf16的，那么这时候通过findmax获取对应的量化参数，然后量化成int8的数值，输入给int8算子。这样整网跑，跑出精度。

如果精度正常，这时候就要开始优化性能了。动态量化有findmax操作，有整个kvcache的量化操作，还要多一份int8 kvcache的临时存储，性能比较差。怎么优化？上静态量化。静态量化的步骤：

（1）按采集的kvcache的分布，找到一组合适的量化参数。

（2）直接把kvcache定义成int8类型，prefill阶段产生的kv值，直接根据量化参数进行量化然后放到kvcache里。

（3）decode阶段，产生的kv值，根据量化参数，量化后再放入kvcache里。

（4）把q量化一下，然后和原本就是int8的kvcache传输给算子做计算就行。

以上步骤可以在PD混合里执行，也可以在pd分离的场景中操作。

# **三、项目执行步骤**

## **（1）选定qwen某个模型，分析kvcache**

可以以qwen 7B或者啥模型，用vllm或者sglang跑个decode过程，采集kvcache，分析kvcache的分布。确定量化维度。

## **（2）用pytorch拼凑出对应的量化pageattention算子**

开发单测，把vllm/sglang里用来跑decode的pageattention算子拿出来，然后用pytorch拼一个量化的pageattention实现，作为int8 kvcache的量化算子。输出的结果是o，和原生的pageattention算子的输出结果，在误差范围内，比如5%或者10%左右就差不多。大模型有一定的容错性，偏差一点，在实际模型里未必会掉精度。

## **（3）在模型中实现动态量化验证精度**

在pageattention那个算子位置，对fp16/bf16的kvcache实现动态量化，先按量化维度findmax获得量化参数，然后用量化公式对kvcache进行量化，q的数值少，可以使用perTensor动态量化，然后输入到pytorch的量化算子里。整个过程都可以用pytorch实现，验好精度后续可以全部用算子实现一遍，以提升性能。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTM3MmY2NDU5OTE4Y2I1OTE3OTgwNzQwNzQxMGIyMWNfYTM5ZGY5OGRiNjYyOGVjZjE1NGM1ZDg3NDkxMDhhZDlfSUQ6NzYxMzA1MTg5ODQwMDE5NzU2M18xNzgyMjE4Nzg1OjE3ODIzMDUxODVfVjM)



## **（4）静态量化优化**

（1）按采集的kvcache的分布，找到一组合适的量化参数。

（2）直接把kvcache定义成int8类型，prefill阶段产生的kv值，直接根据量化参数进行量化然后放到kvcache里。

（3）decode阶段，产生的kv值，根据量化参数，量化后再放入kvcache里。

（4）把q量化一下，然后和原本就是int8的kvcache传输给算子做计算就行。

动态量化精度可以，静态量化也差不多问题不大，这里也不用深究，**只要动态量化完成，静态量化做不做都行，**反正是用来面试的，不是用来生产实践的，能讲清楚就行。没真做过的，也不清楚里面的细节。

## **（5）整体性能优化和测试**

开发int8 kvcache的pageattention算子。对应的开发思路，在算子开发那个项目里写了，此处不再赘述

开发量化算子\+放入kvcache的融合算子；为啥要这么操作？原本实现是放入kvcache的算子，现在要先量化再放入kvcache，多了个量化操作，这就是典型的可以进行算子融合优化性能的思路。所以能融合的就在一个kernel里实现。

开发q的量化算子。

最后端到端的进行精度测试和性能提升测试。

# **三、项目说明**

以上是一个完整的项目体系，里面单独的操作都可以拿出来植入简历里。可以不做完整，只做部分，关键是整体的思路要明确，哪怕你不做，能给面试官讲讲思路也是个加分项。

Int8 kvcache的工程实践难度高，思路摸索当然也难，不过我已经给大家把思路都整理出来了。这个针对不同的模型，需要进行定制化，比如不同的kvcache量化，整个操作可能都会发生变化，但是思路基本就这些。

大家操作以上项目的时候，可以做到动态量化就行了，哪怕精度不行，也可以说行，**反正是面试**，静态量化可以不用干下去，但是思路要掌握，实操这个比较费时间。

感兴趣的可以做下静态量化，也不需要额外开发算子，能把整套链路跑通，就能够有非常多的理解了。最后的整体性能优化可以假装外包给专门开发算子的人干了。



# **四、简历内容填写**

内容：

**负责/参与int8 kvcache整体量化方案设计和实现，包括kvcache分析和量化维度选取，用pytorch搭建量化pageattention算子实现，实现动态量化验证精度可行性。根据动态量化结果，参与设计静态量化方案。整体实现qwen 72B模型的10%性能优化提升。**



说明：这里面故意凸显的坑是：**kvcache分析和量化维度选取，pytorch搭建和实现。以及静态量化。**动态量化相对来说比较好实现，大家去操作实现应该都能在面试中答出来，静态量化细节比较多，大家根据自己掌握的程度要不要去填写。

**根据自己的理解程度挑着写上简历里，最好自己可以再润色一下，防止写的都一样，最后我会在简历上把关一下。**

