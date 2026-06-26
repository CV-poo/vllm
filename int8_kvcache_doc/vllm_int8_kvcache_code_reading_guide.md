# vLLM int8 KV cache 代码阅读指南

目标有两个：

1. 先熟悉 vLLM 推理 infra 的主干：请求如何进入 engine，scheduler 如何安排 token，worker/model runner 如何执行模型，KV cache 如何分配和索引。
2. 聚焦单卡 Qwen2.5-7B 的 int8 KV cache 量化：先读懂 Qwen2 attention、KV cache 写入、paged decode attention，再进入动态量化和静态量化的改动点。

本文按“先建立地图，再深入关键链路，最后看分布式”的顺序组织。建议边读边用 `rg` 搜索、加少量打印或断点验证，不要一上来全仓库通读。

## 0. 先读已有项目文档

重点读：

- `int8_kvcache_doc/kvcache量化项目实操讲解.md`
- `int8_kvcache_doc/4--int8 kvcache性能优化项目.md`

读完后需要记住这条主线：

- KV cache 是按 layer 维护的大块缓存，decode 阶段反复被 paged attention 读取。
- prefill/decode 都会产生新的 K/V，并通过 reshape-and-cache 类算子写入 KV cache。
- 动态量化适合快速验证精度，但会临时量化整段 cache，性能和显存都不理想。
- 静态量化才是工程目标：KV cache 分配时就是 int8，写入时量化，decode attention 直接消费 int8 cache。
- 真正的性能收益主要来自 decode 阶段的 paged attention：减少 KV 访存，并尽量利用 int8 计算。

## 1. 第一阶段：建立 vLLM v1 主干地图

这一阶段以大致读为主。目标不是理解所有优化，而是知道一个请求如何走到模型 forward。

### 1.1 参数和配置入口

大致读：

- `vllm/engine/arg_utils.py`
- `vllm/config/cache.py`
- `vllm/config/attention.py`
- `vllm/config/vllm.py`

重点关注：

- `--kv-cache-dtype`
- `--block-size`
- `--kv-cache-memory-bytes`
- `--gpu-memory-utilization`
- `--calculate-kv-scales`
- `--kv-cache-dtype-skip-layers`
- `--attention-backend`

你需要形成这个认识：KV cache dtype 和 attention backend 都是配置驱动的。要新增一种 int8 静态 cache 类型，不能只改 kernel，还要让配置、dtype 解析、backend selector、KV cache spec 都认识它。

### 1.2 Engine / Scheduler / Worker 的职责划分

大致读：

- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/worker/gpu_model_runner.py`

阅读目标：

- `scheduler` 决定这一轮每个 request 计算多少 token。
- `kv_cache_manager` / `block_pool` 管理哪些 KV block 分给哪些 request。
- `gpu_model_runner` 把 scheduler 输出转换成模型 forward 所需的张量、block table、slot mapping 和 attention metadata。

暂时不用深究：

- LoRA、多模态、Mamba、spec decode、KV transfer、CPU offload。
- 这些都是后续兼容项，单卡 Qwen2.5-7B int8 KV cache 第一阶段可以先跳过。

## 2. 第二阶段：读 Qwen2.5-7B 模型执行链路

Qwen2.5-7B 在 vLLM 中走 Qwen2 代码。

重点读：

- `vllm/model_executor/models/registry.py`
- `vllm/model_executor/models/qwen2.py`
- `vllm/model_executor/layers/linear.py`
- `vllm/model_executor/layers/rotary_embedding.py`
- `vllm/model_executor/layers/attention/attention.py`

### 2.1 Qwen2 模型结构

重点读 `vllm/model_executor/models/qwen2.py`：

- `Qwen2Attention.__init__`
- `Qwen2Attention.forward`
- `Qwen2DecoderLayer`
- `Qwen2Model`
- `Qwen2ForCausalLM`

重点理解 `Qwen2Attention.forward`：

```text
hidden_states
  -> qkv_proj
  -> split q, k, v
  -> rotary_emb(positions, q, k)
  -> self.attn(q, k, v)
  -> o_proj
```

对 int8 KV cache 来说，最关键的是 `self.attn(q, k, v)`。Qwen2 自身只负责产生 fp16/bf16 的 q/k/v，真正写 KV cache 和读 KV cache 做 attention 的逻辑在通用 `Attention` 层和 attention backend 里。

Qwen2.5-7B 通常是 GQA：query heads 多于 kv heads。单卡时 TP size 为 1，`num_heads` 和 `num_kv_heads` 不会被张量并行切分。阅读时先按单卡理解，后面再补 TP。

### 2.2 通用 Attention 层

重点读 `vllm/model_executor/layers/attention/attention.py`：

- `Attention.__init__`
- `_init_kv_cache_quant`
- `Attention.forward`
- `Attention.get_kv_cache_spec`
- `unified_kv_cache_update`
- `unified_attention_with_output`
- `get_attention_context`

这里是整个 KV cache 量化改造的中心入口。

关键逻辑：

1. `Attention.__init__` 根据 `cache_config.cache_dtype` 和模型 dtype 选 backend。
2. `Attention.forward` 把 q/k/v reshape 成 `[num_tokens, heads, head_dim]`。
3. 如果 backend 不在 forward 内部更新 KV cache，则先调用 `unified_kv_cache_update`。
4. 再调用 `unified_attention_with_output`，真正进入 backend 的 `impl.forward`。
5. `get_attention_context` 从 forward context 里取本层的 `attn_metadata`、`kv_cache`、`slot_mapping`。

这一层要着重读，因为动态量化和静态量化都会经过这里。

## 3. 第三阶段：读 KV cache 的规格、分配和索引

这一阶段要着重读。int8 KV cache 改动很大一部分不是 attention 公式，而是 cache 的 dtype、shape、page size、slot 映射和 block table。

### 3.1 KV cache spec 和 quant mode

重点读：

- `vllm/v1/kv_cache_interface.py`
- `vllm/utils/torch_utils.py`
- `vllm/config/cache.py`

重点对象：

- `KVQuantMode`
- `get_kv_quant_mode`
- `is_quantized_kv_cache`
- `AttentionSpec`
- `FullAttentionSpec`
- `AttentionSpec.page_size_bytes`
- `AttentionSpec.real_page_size_bytes`

当前代码里已经有：

- `FP8_PER_TENSOR`
- `INT8_PER_TOKEN_HEAD`
- `FP8_PER_TOKEN_HEAD`
- `NVFP4`

这不等于已经支持你要做的静态 per-head/per-channel int8 KV cache。现有 `int8_per_token_head` 是每个 token、每个 head 动态 scale，并且 scale 被放进 cache 额外 padding 区域。你的目标如果是静态 per-head 或 per-channel scale，需要新增或复用另一套语义。

建议思考：

- 如果做 per-head 静态量化，scale shape 更像 `[num_layers, 2, num_kv_heads]` 或每层两个 `[num_kv_heads]`。
- 如果做 per-channel 静态量化，scale shape 更像 `[num_layers, 2, num_kv_heads, head_dim]`。
- 现有 `_k_scale` / `_v_scale` 默认是标量，直接复用前要确认 shape 和广播语义。

### 3.2 KV block 管理

大致读：

- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/worker/block_table.py`

重点理解：

- block 是 KV cache 的分配单位。
- block table 把 request 的逻辑 token block 映射到物理 KV block。
- slot mapping 把本轮 token 映射到具体 KV cache slot。
- prefix caching 会复用已缓存 block，因此读 `kv_cache_utils.py` 时先理解 block hash 即可，不必深入所有 hash 细节。

暂缓：

- hybrid KV cache、Mamba cache、encoder cache。

### 3.3 KV cache 分配和 reshape

重点读：

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/utils.py`

重点函数：

- `GPUModelRunner.get_kv_cache_spec`
- `GPUModelRunner.initialize_kv_cache`
- `GPUModelRunner._allocate_kv_cache_tensors`
- `GPUModelRunner._reshape_kv_cache_tensors`
- `GPUModelRunner.initialize_kv_cache_tensors`
- `bind_kv_cache`
- `_reshape_attention_kv_cache`

关键认识：

- 原始 raw KV buffer 通常以 `torch.int8` 字节形式分配，再通过 `.view(dtype)` 变成目标 dtype 的 tensor。
- `AttentionSpec.page_size_bytes` 决定每个 block 占多少字节。
- backend 的 `get_kv_cache_shape` 决定逻辑 shape，例如 Flash/Triton 常见是：

```text
[num_blocks, 2, block_size, num_kv_heads, head_size]
```

- backend 的 `get_kv_cache_stride_order` 决定实际物理布局。

对静态 int8 来说，这里必须读透。因为 dtype 改成 int8 后：

- `page_size_bytes` 会变小。
- 同样显存下 `num_blocks` 会变多。
- `get_kv_cache_shape` 是否需要 padding/scale 区域，要由你的量化方案决定。
- 如果 scale 不内嵌在 KV cache，而是 layer buffer，则 page size 不需要为 scale 加额外空间。

## 4. 第四阶段：读 attention backend 和 paged attention

这一阶段是 int8 kvcache 的主战场。

### 4.1 Backend selector

重点读：

- `vllm/v1/attention/selector.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/platforms/cuda.py`

重点理解：

- 默认 backend 由平台、head size、dtype、kv cache dtype、block size 等一起决定。
- Qwen2.5-7B 单卡 CUDA 上很可能默认走 `FLASH_ATTN`。
- 首次做动态量化实验建议强制走 `TRITON_ATTN`，因为 vLLM 自己的 Triton 路径更容易读、改、加 PyTorch reference。

运行实验时可以关注 `--attention-backend TRITON_ATTN`。如果你只是读代码，可以先读 FlashAttention 和 Triton 两条路径，但动手建议从 Triton 开始。

### 4.2 FlashAttention 路径

重点读：

- `vllm/v1/attention/backends/flash_attn.py`

重点函数：

- `FlashAttentionBackend.get_kv_cache_shape`
- `FlashAttentionBackend.get_kv_cache_stride_order`
- `FlashAttentionImpl.forward`
- `FlashAttentionImpl.do_kv_cache_update`

读法：

- 大致读 metadata builder。
- 着重读 `forward` 里如何从 `kv_cache.unbind(1)` 得到 `key_cache` 和 `value_cache`。
- 着重读 `do_kv_cache_update` 如何调用 `reshape_and_cache_flash`。
- 注意 quantized KV cache 时，目前主要是 fp8 语义：`key_cache.view(current_platform.fp8_dtype())`，并用 `_k_scale/_v_scale` descale。

FlashAttention 路径适合了解默认生产路径，但不建议作为第一版 int8 PyTorch 原型的入口。

### 4.3 Triton attention 路径

重点读：

- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/attention/ops/triton_decode_attention.py`
- `vllm/v1/attention/ops/triton_prefill_attention.py`

重点函数：

- `TritonAttentionBackend.get_kv_cache_shape`
- `TritonAttentionImpl.forward`
- `TritonAttentionImpl.do_kv_cache_update`
- `triton_reshape_and_cache_flash`
- `triton_reshape_and_cache_flash_per_token_head_quant`
- `unified_attention`
- `kernel_unified_attention`

这里要着重读现有 `int8_per_token_head` 路径：

- `TritonAttentionBackend.supported_kv_cache_dtypes` 已包含 `int8_per_token_head`。
- `get_kv_cache_shape` 对 per-token-head scale 在 head dim 后面加 padding。
- `TritonAttentionImpl._ensure_scale_caches` 从 cache padding 里 carve 出 scale view。
- `do_kv_cache_update` 用 `triton_reshape_and_cache_flash_per_token_head_quant` 写 cache。
- `forward` 把 `k_scale_cache/v_scale_cache` 传给 `unified_attention`。
- `kernel_unified_attention` 根据 `KVQuantMode` 选择 dequant/scale 逻辑。

这条路径是你理解“如何把一种新 KV quant mode 接入 vLLM”的最好参考。

## 5. 第五阶段：读测试，建立 reference 思维

这一阶段要着重读。做算子和量化，不读测试很容易误解语义。

重点读：

- `tests/kernels/attention/test_attention.py`
- `tests/kernels/attention/test_triton_decode_attention.py`
- `tests/kernels/attention/test_triton_unified_attention.py`
- `tests/kernels/test_cache_kernels.py`
- `tests/quantization/test_per_token_kv_cache.py`
- `tests/models/quantization/test_per_token_kv_cache.py`
- `tests/v1/attention/test_attention_backends.py`
- `tests/v1/attention/test_attention_backends_selection.py`

最重要的是 `tests/kernels/attention/test_attention.py` 里的 reference：

- `ref_masked_attention`
- `ref_single_query_cached_kv_attention`

它展示了 paged attention 的基本语义：

1. 根据 block table 找到每个 request 的物理 block。
2. 根据 token offset 从 key/value cache 取历史 KV。
3. GQA/MQA 时 repeat KV heads。
4. 做 `q @ k -> softmax -> p @ v`。

建议你先把这套 reference 单独理解透，再谈 int8。int8 pageattention 的 PyTorch 等效实现就是在这套逻辑上加 quant/dequant。

## 6. 第六阶段：单卡动态 int8 KV cache 阅读与实验路线

这是你的第一阶段动手目标：不改 KV cache 存储 dtype，先验证 int8 pageattention 数值是否可接受。

推荐顺序：

1. 强制 `TRITON_ATTN`，让路径固定。
2. 先用现有 fp16/bf16 KV cache 跑通 Qwen2.5-7B 的短 prompt decode。
3. 在 `TritonAttentionImpl.forward` 或独立测试里拿到：
   - `query`
   - `key_cache`
   - `value_cache`
   - `block_table`
   - `seq_lens`
   - `query_start_loc`
4. 基于 `tests/kernels/attention/test_attention.py` 写 PyTorch reference。
5. 在 reference 里加入 int8 quant/dequant。
6. 对比原始 backend 输出和 int8 reference 输出。
7. 再跑端到端 generation，看是否乱码、困惑度/benchmark 是否明显下降。

动态量化建议先读和实验这些量化方式：

- per-tensor：最简单，只用于 sanity check。
- per-head：scale shape 大致为 `[num_kv_heads]`。
- per-channel：scale shape 大致为 `[num_kv_heads, head_dim]`。

重点注意：

- per-channel 下 scale 往往不能简单从 `q * k` 里提成公因式。
- 如果希望未来 int8 kernel 数学等效，需要提前设计好 scale 放在哪里、什么时候量化 q。
- 你文档里提到的“让 q 先乘 k 的 scale，再对乘积后的 q 做量化”需要重点验证。

动态阶段建议大致读：

- config 接入
- engine/scheduler
- KV cache 分配

动态阶段必须着重读：

- `Qwen2Attention.forward`
- `Attention.forward`
- `TritonAttentionImpl.forward`
- `triton_unified_attention.py`
- attention tests 中的 reference。

## 7. 第七阶段：单卡静态 int8 KV cache 阅读与设计路线

静态量化才是完整工程目标。建议在动态量化确认可行后再读和设计。

### 7.1 需要新增或改动的框架位置

重点读并记录改动点：

- `vllm/config/cache.py`
  - 新增 cache dtype 字符串，例如 `int8_static_head` 或 `int8_static_channel`。
- `vllm/utils/torch_utils.py`
  - dtype string 到 torch dtype 的映射。
  - `is_quantized_kv_cache` 相关路径。
- `vllm/v1/kv_cache_interface.py`
  - 新增 `KVQuantMode`。
  - 更新 `get_kv_quant_mode`。
  - 确认 `AttentionSpec.page_size_bytes` 是否需要包含 scale 存储。
- `vllm/model_executor/layers/attention/attention.py`
  - scale buffer 的注册、加载、默认值、shape。
  - `Attention.get_kv_cache_spec` 生成正确 spec。
- `vllm/v1/attention/backends/triton_attn.py`
  - backend 支持新 dtype。
  - `get_kv_cache_shape`。
  - `TritonAttentionImpl.forward`。
  - `TritonAttentionImpl.do_kv_cache_update`。
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
  - 写入 KV cache 前量化 K/V。
- `vllm/v1/attention/ops/triton_unified_attention.py`
  - decode 读 int8 cache 并按你的 scale 语义 dequant 或 int8 计算。

### 7.2 静态方案的关键设计问题

读代码时带着这些问题看：

1. `kv_cache.dtype` 应该是 `torch.int8` 还是 raw int8 view 后再解释？
2. scale 存在哪里？
   - layer buffer：实现简单，page size 不变。
   - cache padding：适合 per-token-head，不一定适合静态 per-channel。
   - 独立 scale tensor：更清晰，但要接入生命周期和 CUDA graph。
3. scale 是每层一份，还是 K/V 分开？
4. scale 是否参与 checkpoint 加载？
5. 如果没有 scale checkpoint，是否允许 runtime 从校准文件加载？
6. prefill 写 cache 时如何量化 K/V？
7. decode 新 token 写 cache 时如何复用同一套量化逻辑？
8. chunked prefill 从 int8 cache 读历史 KV 时是否要反量化再走 prefill attention？
9. CUDA graph capture 中是否会出现动态 shape、动态 scale 分配或 Python 分支？

### 7.3 静态实现的读码顺序

建议顺序：

1. `CacheConfig.cache_dtype` 和 CLI 参数，让 vLLM 能接受新 dtype。
2. `get_kv_quant_mode` 和 `kv_cache_dtype_str_to_dtype`，让 dtype 能变成 `torch.int8`。
3. `Attention.get_kv_cache_spec`，让 KV cache spec 的 dtype/page size 正确。
4. `GPUModelRunner._allocate_kv_cache_tensors` 和 `_reshape_kv_cache_tensors`，确认实际分配出来的 cache shape。
5. `TritonAttentionImpl.do_kv_cache_update`，实现写入时量化。
6. `TritonAttentionImpl.forward` 和 `triton_unified_attention.py`，实现读取时 int8 attention。
7. tests，先单测再端到端。

不要一开始就改 FlashAttention。先在 Triton 路径跑通逻辑，再考虑高性能 kernel 或 FlashAttention 对接。

## 8. 建议的调试和验证顺序

### 8.1 最小链路验证

先做小模型/短输入，固定随机种子和 backend：

1. Qwen2.5-7B 单卡能正常 generate。
2. 强制 `TRITON_ATTN` 后仍能 generate。
3. dump 一层的 `query/key_cache/value_cache/block_table/seq_lens`。
4. PyTorch reference 输出对齐原始 Triton/FlashAttention 输出。
5. 动态 int8 reference 输出和 fp16 reference 做误差统计。
6. 接入动态 int8 路径，看端到端输出是否稳定。

### 8.2 精度验证

建议逐层推进：

- 单层 attention 输出误差。
- 全模型短 prompt 文本稳定性。
- 固定数据集少量样本。
- AIME/HumanEval/GPQA 等正式 benchmark。

动态量化确认精度后，再做静态量化。否则静态链路出错时很难判断是 scale 不好、cache 写错、block table 错，还是 attention kernel 错。

### 8.3 性能验证

静态 int8 才值得做性能测试。重点看：

- decode attention kernel 时间。
- TPOT。
- 长上下文下的吞吐变化。
- 相同显存下 `num_gpu_blocks` / max concurrency 是否上升。

动态量化通常不适合证明性能收益，因为它多了 findmax、整 cache 量化和临时 int8 cache。

## 9. 哪些大致读，哪些着重读

着重读：

- `vllm/model_executor/models/qwen2.py`
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/gpu_model_runner.py` 中 KV cache 初始化、attention metadata、slot mapping 相关函数
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/utils.py` 的 `bind_kv_cache`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`
- `vllm/v1/attention/ops/triton_unified_attention.py`
- `tests/kernels/attention/test_attention.py`
- `tests/kernels/attention/test_triton_decode_attention.py`

大致读：

- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/engine/arg_utils.py`
- `vllm/config/cache.py`
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/v1/attention/backends/flash_attn.py`

最后再读：

- 分布式并行相关：TP/PP/DP/DCP/CP。
- KV transfer / PD 分离。
- Speculative decoding。
- CUDA graph 深度兼容。
- Prefix caching 的全部 hash 和 eviction 细节。

暂时跳过：

- 多模态。
- LoRA。
- Mamba/hybrid 模型。
- MoE。
- CPU/XPU/ROCm 特殊路径。
- DeepSeek MLA，除非后续要做 MLA int8 KV cache。

## 10. 多卡和分布式内容最后再看

单卡跑通后，再按下面顺序补：

### 10.1 Tensor Parallel

重点读：

- `vllm/distributed/parallel_state.py`
- `vllm/model_executor/models/qwen2.py` 中 `get_tensor_model_parallel_world_size`
- `QKVParallelLinear`
- `RowParallelLinear`
- `ModelConfig.get_num_attention_heads`
- `ModelConfig.get_num_kv_heads`

关键问题：

- Q heads 和 KV heads 如何按 TP 切分或复制。
- 每张卡上的 KV cache scale 是本地 heads 的 scale，还是全局 heads 的切片。
- per-channel scale 在 TP 下的 shape 如何定义。

### 10.2 Pipeline Parallel / Data Parallel

大致读：

- `get_pp_group`
- worker 初始化和 model runner 创建逻辑。
- scheduler 和 worker 之间的 batch 协调。

关键问题：

- 每个 PP stage 只持有部分 layers，因此每个 stage 只需要本地 layers 的 KV scale。
- DP 每个 replica 独立维护 KV cache，scale 可以相同，但 cache 内容不同。

### 10.3 Context Parallel / Decode Context Parallel

最后读：

- `vllm/v1/worker/cp_utils.py`
- `get_dcp_group`
- `get_dcp_local_seq_lens`
- FlashAttention/Triton forward 中 DCP 分支。

关键问题：

- 长上下文被切到不同 rank 后，每个 rank 看到的 KV token 子集不同。
- 静态 scale 如果来自全量校准，需要确认各 rank 使用一致的 scale 切片。

### 10.4 PD 分离 / KV transfer

最后读：

- `vllm/distributed/kv_transfer`
- `vllm/v1/worker/kv_connector_model_runner_mixin.py`
- `vllm/v1/worker/gpu/kv_connector.py`

关键问题：

- int8 cache 传输能节省带宽。
- 接收端必须知道 dtype、layout、scale 语义。
- 如果 scale 不在 cache 内，scale 也必须同步或可重新加载。

## 11. 推荐实践路线

建议按下面顺序推进：

1. 读 Qwen2 + Attention + TritonAttention，画出单卡 forward 数据流。
2. 读 KV cache spec + allocation + reshape，确认 cache shape 和 block/slot 语义。
3. 读 attention tests，自己写一个最小 PyTorch paged attention reference。
4. 在 reference 里加动态 int8 quant/dequant，做单层误差对齐。
5. 端到端接动态 int8，先验证精度可行性。
6. 统计 Qwen2.5-7B KV cache 分布，比较 per-head 和 per-channel。
7. 设计静态 scale 格式和 cache dtype。
8. 在 Triton 路径实现静态 int8 写 cache 和读 cache。
9. 跑单测、短文本、数据集精度。
10. 再考虑性能 kernel、FlashAttention 路径和多卡。

如果读代码时只抓一条链路，请抓这条：

```text
Qwen2Attention.forward
  -> Attention.forward
  -> unified_kv_cache_update
  -> TritonAttentionImpl.do_kv_cache_update
  -> triton_reshape_and_cache_flash

  -> unified_attention_with_output
  -> TritonAttentionImpl.forward
  -> unified_attention
  -> kernel_unified_attention
```

这条链路就是 int8 KV cache 单卡原型最核心的代码路径。
