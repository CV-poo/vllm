# kvcache量化项目实操讲解

**动态量化和静态量化执行思路**

# **1、kvcache的流程**

Kvcache是啥？是个连续的大tensor，shape是\[block，block\_size，2，head，head\_dim\]，整个推理引擎启动后，所有的用户和所有的请求都共用这一份kvcache。head\*head\_dim就是一个token的特征向量，就是指代一个token，从token映射到向量的表征。2是啥，分别是k和v。block\*block\_size是啥？是指能容纳的token数量，kvcache就是装token的kv特征向量的大tensor。有专门一套机制来管理这个kvcache。block\_size设置的太大会浪费，装token时可能会装不满，装不满那不就浪费了吗，设置的太小会影响pageattention算子的性能，因为你在decode阶段做注意力机制计算时，要从kvcache里拿属于自己的kv值，跨block去拿，那不就不连续了吗，离散访问了，那访存效率肯定不如在一个block内拿的效率高。所以block\-size不能太大也不能太小，一般就32/64/128/256这种。那shape里除了block没定，其它数值都定了，block怎么定？看你的hbm大小呗，除去一些固定的开销比如模型权重占用、cudagraph私有存储，剩下的就都给kvcache了，也就都体现在block大小上了，越大你的推理服务能承受的并发数就越大（就是用户同一时刻给你发的请求，也就是向你问的问题），那肯定我并发数越大对我云服务商来说越赚啊，几台机器服务全世界，电费就那么多，印钞机一样。这就是量化的好处，**算的快，省存储**。

Kvcache在vllm/sglang里的流程是啥？也就是哪里需要用到kvcache：

Prefill阶段产生kv值，怎么产生？和权重相乘产生。产生后要存到kvcache里，怎么存？用reshape\_and\_cache类似的算子，把k和v快速存到kvcache的tensor里，说白了就是transpose类型的算子。如果有chunk\_prefill使能，也就是要分段做prefill，那么prefill阶段还要从kvcache里拿kv值，再做prefill。**Prefill这里使用的attention计算，不做量化**，因为attention计算在不开启chunk\_prefill的时候，输入参数是k和v，并不是kvcache。开启了chunk\_prefill，那么输入参数就包括k和v以及kvcache，因为要从kvcache拿其它的kv值。L

PD混合，kvcache就p和d共享的，就直接到decode阶段直接用就行，都在同一台机器上，用的都是同样的卡。Pd分离，不在同一个机器上，那么就需要把kvcache从p的服务器，传给d的服务器。那量化后，这里传输的数量是不是少了，这也是量化的好处之一。

Decode阶段，使用kvcache，怎么使用？**pageattention算子要使用，量化就是对这个pageattention算子做优化，用低精度kvcache，访存提升了，算力提升了，端到端推理性能不就上去了，这就是kvcache量化的最大性能收益来源。**

Decode阶段另外还要把新增的k和v存到kvcache， k和v为啥会新增，新增多少？因为会一个接一个的产生token，新增的是这些新产生的token的k和v，新增的量很少，不就是batch个token。怎么存？跟prefill阶段的存储逻辑一样，把kv值塞到kvcache里，所以一样会用到reshape\_and\_cache类似的算子。

如果有推测解码特性，还会涉及到draft模型的kvcache处理。

以上就是整体推理过程涉及到kvcache的地方，kvcache只是推理过程中的一部分，不是全部，还有很多其它的优化和特性。



# **2、动态量化**

## **（1）为什么要做动态量化？精度的考量**

已知量化有性能提升和存储优化的好处，那要使能这个特性，最重要的是啥，是精度。任何脱离精度说性能都是扯犊子，精度决定了能不能用，性能才是好不好用。

那我们要做kvcache int8量化，不确定量化之后精度是否被接受，所以得以最高效的方法先去验证这条技术路线是否可行对吧。那怎么判断精度？业界通用的黄金标准，就是跑数据集，看数据集的跑分，比如human、gpqa、aime等等。你看量化前和量化后，到底跑分掉了多少，人家费几千万上亿的资金训练一个模型跑分也就提升几个点，你搞个量化掉好些点，那这量化就没意义了。那既然只是做精度验证，肯定执行层面越高效越好啊，python/pytorch代码层面不需要编译，算子需要开发测试编译，肯定能在python/pytorch层级做精度验证是最合适的也是最高效的。所以这就是为啥要先做动态量化的根本原因，快速的进行精度校验，验证量化路线的可行性。

## **（2）pageattention int8量化算子的处理**

之前也分析了kvcache的流程，真正会用到kvcache做计算的只有decode阶段的pageattention和prefill阶段的chunk\_prefill。这里暂不考虑prefill阶段的chunk\_prefill，只考虑decode阶段的pageattention。

我们要做动态量化验证精度，肯定得有对应的pageattention int8量化算子吧，但是开发kernel太费时费力了，要是在pytorch层面有个对应的等效实现就不用编译直接可用了。所以目标就是要获得pageattention int8量化算子的等效实现，且保证它的准确性，你不能写个自己都不确定准不准的算子来糊弄。怎么搞？

原始的pageattention算子是fp16和bf16的kernel实现，vllm里肯定精度是正确的，毕竟大家都在用。那么最开始它这个算子是怎么来的，为啥它调用到模型里，推理结果就是正确的？这个之前在文档里都讲过，最开始本来没有这些融合算子，都是原始的native的pytorch拼凑实现，后面为了加速，才搞出来了融合算子。那融合算子的精度怎么保证？有native实现，跟它精度对齐，算子的精度就能保证了，native的端到端推理结果正确，那么融合算子的精度对齐，端到端推理结果没理由不正确，除非你算子写的有问题内存踩踏了。所以任何融合算子的出生地，也就是它们自己的仓库里，必然会有单元测试用例，这是软件代码开发的规范和标准流程，你那必然能获得对应的native的pytorch等效实现。**大家一定要重试单元测试用例，干过活的人就知道，单测都是宝藏，特别是在算子开发领域，单测就是理解算子语义（算子的功能和数学计算流程）的黄金资料。**

那么到int8 pageattention算子这里了，首先我们得保证算子的正确性，这是核心的核心，那按上边的逻辑，开展单元测试就是检验算子正确与否的黄金标准。那关键怎么单元测试呢？你得有个等效的pytorch实现或者c\+\+实现，问题又来了，你有吗？你没有，那咋办？自己实现呗。那自己搞的等效实现你又怎么判断它的正确性呢？在单元测试这一层级已经没法验证了，只能在模型里端到端测试，这是更高一级的集成测试，如果我放在模型里测试都没问题了，那单测肯定就没问题了，你就成为了单测的标准。

怎么判断都搞清楚了，接下来就是实现了。因为量化就是对输入的q和kvcache做了量化，计算流程是基本不变的，所以在计算流程层面跟原始的fp16/bf16实现具有高度的一致性，所以比较好保证正确的就是基于原始的fp16/bf16实现进行修改，毕竟有个精度标杆的实现，要增加额外的处理就是数值精度的处理，别的别乱动肯定就能最大程度的保证开发的正确性，再不济你也好debug啊，毕竟改的代码就只有数值精度那里。

但是原始的实现只有kernel，所以搞到原始实现的等效fp16/bf16 pytorch实现，再基于该实现，进一步处理数值精度，改成int8的等效pytorch实现，再进行模型端到端验证确认正确性，最后基于int8的等效pytorch实现（也是单测的精度标杆），再对照原始的fp16/bf16的kernel实现，改出一版int8的kernel实现，问题不就完美解决了吗。

## **（3）获取原始fp16/bf16的pageattention的pytorch等效实现**

attention有太多的算子变种，语义非常复杂，你稍微哪里搞的不对，写的就是错的。虽然现在ai比较强大，根据kernel实现让它搞个pytorch实现是比较容易的，但是attention不一样，语义太复杂了，你搞一天可能都会在ai幻觉中度过。那么这时候原生的测试用例就是宝藏了。

这里原始的fp16/bf16的pageattention的kernel实现是能基于vllm代码找到的，找到kernel在哪，就能找到对应的测试用例在哪，找到测试用例在哪就能从测试用例里扒出来对应的等效实现。通过构建单元测试用例，来确保你扒出来的等效实现是正确的。这些就可以让ai给你搞定了，把kernel和对应的测试用例给ai展示，让它扒出一个等效实现就完事了。

所以这里核心是怎么看代码，需要自己去锻炼一下，通过实践的方式就是加打印，确认pageattention到底走的哪里实现。Vllm里会跨仓库调用，现在最新的vllm0\.16\.0，调用的是flash\-attn的仓库里的算子来做pageattention，大家去实践的时候看看。

原始fp16/bf16的pageattention的pytorch等效实现，以及对应的测试用例我已经弄好了。希望大家能自己再去实操一下，看自己能不能搞出来，实操成功，你就学到了，不然学的还是皮毛，infra领域，光看不动手会吃巨亏的。表扬一些同学，实打实的就是干，到面试那就是乱杀，也确实在乱杀，面评都比较好，相比不动手干的就是降维打击。工程实践领域，勤奋动手实践\+思考，你就能成为顶级专家。

## **（4）基于fp16/bf16的pageattention的p****ytorch等****效实现改写成int8的实现**

有了原生的等效实现，后续流程就简单了。量化相关的知识内容在三四讲里。

在pageattention计算那个地方，传入kvcache之前，把q和kvcache做量化，会返回量化参数和int8量化的q和kvcache。伪代码如下：

q\_int8, q\_scale = int8\_quant\(q\)

kvcache\_int8, kvcache\_scale = int8\_quant\(kvcache\)

注意以上int8\_quant是需要选用量化方式，有per\_head量化，对应的量化参数的shape就是\[head\]。per\_channel量化，对应的量化参数就是\[head, head\_dim\]。理论上per\_head量化精度可以的话，per\_channel量化百分百可以，因为量化粒度更细，量化误差肯定更小。怎么选量化方式？分析kvcache的分布，看在哪个维度上越集中，在哪个维度上做量化肯定精度损失最小。为啥不做per\_tensor或者per\_token量化，per\-tensor量化是整个kvcache只有一个量化参数，这显然量化损失会大，万一出现个异常值，整个kvcache量化结果就完犊子了，精度全损失了。per\_token量化，不适合做静态量化，只能用来做纯动态量化，因为跟token相关，你新增个token就得量化一下，量化参数也会发生变化，而且性能会差。所以perchannel量化是最适合的，可以做静态量化，因为量化参数不随token数量发生变化，量化参数就能事先固化下来。



量化结果然后传入：

int8\_pageattention\_pytorch（q\_int8, q\_scale, kvcache\_int8, kvcache\_scale）

注意int8\_pageattention\_pytorch里的实现了，要怎么改？注意核心的注意力机制计算，是p\_ = q\*k，p = softmax（p\_），o = p\*v。

先看第一个，q和k现在都是int8的，跟原始的数学等效的话，肯定是要先反量化再相乘，用自己的scale反量化，再相乘。如下：

q\_int8 \* q\_scale \* kvcache\_int8 \* kvcache\_scale

但是这里pytorch的aten算子里是没有int8的算子，只有fp32和fp16和bf16的算子，应该还对接fp8的算子。所以你没法做int8的计算，只能用模拟的方式，如下：

q\_int8\.float\(\) \* k\_int8\.float\(\) \* q\_scale \* kvcache\_scale

这样模拟int8 kernel计算，将来自己写kernel时，也一样的处理，不要转到float了，才能用到int8的算力。注意这里的计算结果要搞成fp32或者fp16的，fp16可以加速，因为下一个就是softmax了，不过fp16可能会溢出，pytorch层用fp32吧，kernel那里可以用fp16或者bf16加速冒险一下。

softmax之后的p是fp32的假设，跟int8的v做乘积怎么做？将v反量化，然后做计算，然后把输出转回fp16/bf16。这样不就完成了整个的计算流程。

o =（ p\_fp32\*v\_int8\.float\(\)\* kvcache\_scale）\.to\(fp16/bf16\)

当然将来为了int8算力计算，可以对p\_fp32在kernel里动态量化成int8，然p和v的gemm在int8下加速计算。模拟的话就如下：

p\_int8， p\_scale = quant\(p\_fp32\) 

o =（ p\_int8\*v\_int8\.float\(\)\* kvcache\_scale \* p\_scale）\.to\(fp16/bf16\)

最终性能加速还是要在int8下做。

以上实现可以借助ai完成，然后理解相关的代码。



以上per\_head和perchannel最好都实现并跑一下精度，**注意上面只是伪代码，量化参数是否可以提取作为公因式是要判断的。**然后对比下跑数据集的精度，要求跑下aime、human的，对比两种量化方式，以及和原始的跑分。用qwen系列的模型就行，1\.5b\~7b，单卡跑。跑好结果可以写进自己的简历里。

特别注意：

有同学实践下来发现：q\_int8\.float\(\) \* k\_int8\.float\(\) \* q\_scale \* kvcache\_scale。此处perchannel的方式下scale无法作公因式提取，导致没法让q和k做int8的乘法，会在数学上无法等效，怎么解决呢？

**让q先乘上k的scale，然后再对乘积后的q做量化，此时就能数学等效了。**



# **3、静态量化**

## **（1）为什么要做静态量化？性能\+存储**

首先回顾下动态量化的缺点，kvcache还是fp16的，用int8的pageattention算子时，需要把fp16的kvcache量化成int8的，原来的kvcache还在，新的kvcache虽然是中间量，但是会增加峰值显存，何况kvcache本身占用巨大，多了半份的存储，那可支持的并发数会大大降低。

另外每次都要对整个kvcache执行量化，性能也会差。

静态量化的好处，首先一上来kvcache初始化时的dtype就是int8，注意由于存储相比fp16的小，推理引擎会把可用的全部HBM都给kvcache，也就意味着int8的kvcache的block数量会是原fp16的kvcache的block数量的两倍。可容纳更多的token，也就意味着并发数可以大幅提升。

另外的话就是性能大幅提升。先来分析静态量化有哪些需要改动的，以及相比fp16额外的操作有哪些。

## **（2）静态量化流程以及vllm框架的改动。**

假如已经有了一份静态的量化参数，框架层的改动如何：

（a）kvcache的dtype修改为int8，要在vllm源码里改动。

**Prefille阶段：**

（b）此时kvcache\_int8是int8的，prefill阶段的kv值仍然是fp16/bf16，注意激活值的数据流一直都是fp16/bf16的类型。那么把fp16/bf16的kv值通过 reshape\_and\_cache塞到int8的kvcache\_int8必然报错，所以此处要干啥？先量化再塞入，伪代码如下：

scale = get\_kvcache\_scale（layer\_id） \#每层都有kvcache。

k\_int8, v\_int8 = perchannel\_quant\(k，v, scale\)  \#量化，**可用pytorch**

reshape\_and\_cache（k\_int8, v\_int8，kvcache\_int8） \#塞入kvcache，kernel可能不支持int8，**可用pytorch实现，纯数据流操作。**

**注意实际kernel开发时，量化和reshape\_and\_cache可以做算子融合，等于不增加额外耗时。**

（c）如果开启chunk\_prefill，需要熟悉chunk\_prefill的原理

需要从kvcache\_int8里拿过往chunk的kv值，但是kvcache\_int8是int8的数值，kv值得是fp16的，才能做fp16的fa，所以得有个反量化的过程。

k\_int8, v\_int8 = get\_kvcache（kvcache\_int8）

scale = get\_kvcache\_scale（layer\_id）

k，v = perchannel\_unquant（scale，k\_int8, v\_int8）\#反量化，可用pytorch实现，这里要和vllm原始的逻辑对齐，可能不止这个操作，否则会推理结果错误。

**Decode阶段：**

（d）pageattention的int8计算，已经有了int8\_pageattention\_pytorch实现，最后再开发kernel。

获取量化参数，量化q，传入int8\_pageattention\_pytorch:

scale = get\_kvcache\_scale（layer\_id）

q\_int8, q\_scale = perchannel\_quant q\)   \#pytorch实现q的动态量化，q不做静态量化

int8\_pageattention\_pytorch（q\_int8, q\_scale, kvcache\_int8, scale）

（e）新增的kv值塞入kvcache中

这个同prefill的塞入操作

scale = get\_kvcache\_scale（layer\_id） \#每层都有kvcache。

k\_int8, v\_int8 = perchannel\_quant\(k，v, scale\)  \#量化，**可用pytorch**

reshape\_and\_cache（k\_int8, v\_int8，kvcache\_int8） \#塞入kvcache，kernel可能不支持int8，**可用pytorch实现，纯数据流操作。**

## **（3）精度校验**

以上是框架层的大致改动，都可以用pytorch实现来改动，接下来是精度校验。

（1）首先你得获得一份静态量化参数。单卡跑个数据集，最好选用aime数据集，纯数学题，比较客观。dump出kvcache，然后看数值分布，选择合适的参数作为量化参数，比如最大值，或者分位值。

（2）将量化参数结合以上框架层的改动，端到端跑下数据集，看下数据集精度如何。如果精度下降的厉害或者乱码，可能框架层的改动有问题，或者量化参数有问题，需要debug？

怎么debug？首先确认kvcache量化参数是否有误，怎么判断，去你的最开始做的动态量化那里，把动态量化kvcache的过程，换成load你的静态量化参数，然后对kvcache做量化，这样跑数据集如果精度不行，说明你的静态量化参数有问题，如果精度没问题，说明是你的框架层改动有问题。

## **（4）静态量化的性能分析，必须要掌握性能收益的来源**

分析一下静态量化后最终的性能提升来源：

**原始的p端的流程：**

Prefill：fp16kvcache的reshape\_and\_cache—》fp16 kvcache传输

Chunk\_prefill: 从fp16kvcache获取fp16 kv值。

**Int8的p端的流程：**

Prefill：int8 kvcache的量化\+reshape\_and\_cache的融合—》int8 kvcache传输

Chunk\_prefill: 从int8kvcache获取int8 kv值\+反量化。

可以看到，prefill阶段从纯数据流的fp16算子，变成量化\+int8数据流的算子，性能几乎没啥差异。Kvcache传输int8占便宜，时间是会摊到tpot的，不计入ttft的时间，为啥？因为首字token的生成和kvcache无关，它两互不依赖。



**原始的d端流程：**

Decode：fp16的pageattention \+ fp16的reshape\_and\_cache

**Int8的d端流程：**

Decode：int8的pageattention \+ （int8的reshape\_and\_cache \+ kv量化的算子融合）

可以看到，decode阶段，reshape\_and\_cache的差异几乎可以忽略不计。int8的pageattention是性能提升来源，也是这项技术的提升来源，所以int8的pageattention kernel实现至关重要，算子的性能提升有多大，你整体tpop的提升就能有多大。算子的理论提升来源于哪里？int8访存\+int8计算，所以文本越长的推理，你的提升越大。所以涉及gemm计算的，要用tensorcore，要用int8的算力，也就意味着p和v乘那里对p又得在kernel里做一次动态量化，softmax那里可以用低精度来极限加速。



## **（5）静态量化的完整开发，兼容各项特性**

**（a）dump量化参数，要考虑数据集和考虑分布式并行，考虑方法论**

静态量化的核心是啥，是量化参数是否合理。

量化参数是否能适配不同的数据集，这才是最关键的。给出参考判断，通过aime校准的量化参数，精度几乎无损。那问题来了，怎么获取aime的量化参数。

单卡时怎么处理？

分布式并行时怎么处理？数据并行\+张量并行\+专家并行，所以要熟悉分布式并行对这里的影响。才能处理好分布式多卡推理下的推理精度。

量化参数从kvcache里拿出来时怎么处理？最大值还是分位值？

**（b）cudagraph的兼容**

可以无缝兼容cudagraph吗？需要额外操作啥吗？把cudagraph的原理和vllm里的代码好好看下，分析一下。

**（c）推测解码的特性**

推测解码使能后，除了target模型的kvcache要量化，draft的kvcache怎么处理，跟着一起量化吗？如果不跟着一起量化呢？怎么在vllm代码里做好隔离，怎么实现。

对接受率的影响如何？对性能加速的效果影响如何？

把推测解码的原理好好搞清楚。然后把这个兼容到推测解码里。

**（d）验证pd混合和pd分离场景下的精度是否均正常**

把pd分离原理搞清楚，支持两种场景。

并且分析pd分离时，高并发下，ttft和tpot在使能int8 kvcache特性前后的差异。

**（e）量化兼容性分析**

比如现在是w16a16c8量化，能否兼容到w8a8量化，以及能否兼容到w4a8量化，为什么？

把w和a的量化搞清楚，也可以实际的操作一下。





围绕以上特性，把对应的特性都学清楚，把对应的问题都解决。



