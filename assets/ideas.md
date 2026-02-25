## conditions
### must-have
- high efficiency
- autoregressive training/inference
- discrete concept tokens
- minimal structural changes to the base LLM (LoRA + small new modules)

### preferred
- internal concept tokens only (not exposed to user text)
- online compression during decoding

## v1
- Kept as baseline in `train_qwen.py`.

## v2: Online Concept Compression with Concept-Only Middle Blocks

### core design
- Split decoder layers into:
  - shallow blocks: process all normal tokens
  - middle blocks: process concept tokens only
  - deep blocks: process concept prefix + dynamic normal tail
- Add a `ConceptHead` on top of shallow hidden states:
  - output size = `4096 + 1` (`4096` concept ids + `<NULL>`)
  - use Gumbel-Softmax with straight-through estimator
  - `<NULL>` means current normal token is not represented as concept token

### online boundary
- For each time step `t`, define:
  - `tau(t)`: most recent compressed normal token position (`!= <NULL>`)
  - `tail(t) = h_shallow[tau(t)+1 : t]`
- Deep input at step `t`:
  - `Z_t = [concept_prefix(<=t); tail(t)]`

### deep attention rules
- Concept -> Concept: allowed
- Tail -> Concept: allowed
- Tail -> Tail: causal
- Concept -> Tail: disallowed

### training objective (current version)
- Pure next-token cross entropy only:
  - `CE(logits_t, x[t+1])`
- No extra auxiliary losses and no additional training tricks in this version.

### trainable parts
- New modules:
  - `ConceptHead`
  - `ConceptEmbedding`
- Base model:
  - LoRA adapters on attention/MLP linear layers
- Base backbone weights are frozen except LoRA parameters.

### implementation file
- `train_qwen_2.py`


### 实现的tricks
1. 控制NULL tokens的数量
2. 控制连续的NULL tokens的数量
   
### 优势
1. 相比于LLM as compressor and decompressor，可以把生成的内容也压缩成latent tokens，这样与concept tokens的故事可能融合得更好
2. E2E

---


---

### prompt
好，那我直接给你一份**“极简 + 伪代码级”的 coding agent prompt**，基本是你脑子里那套逻辑的**可执行规格**，尽量减少自由发挥空间。

---

#### Ultra-Compact Coding Agent Prompt（Pseudo-code Style）

**Task**
Implement training code for an autoregressive LM with **online concept compression** and **concept-only middle Transformer blocks**.

---

##### Core Idea

* Normal tokens are processed by **shallow layers**.
* After shallow layers, a **concept head** compresses some tokens into **concept tokens** using **Gumbel-Softmax + straight-through**.
* **Middle Transformer blocks run ONLY on concept tokens** (main compute saving).
* **Deep Transformer blocks** see:

  * all concept tokens
  * * a dynamic tail of normal tokens (from last compressed token + 1 to current token)
* Model predicts next normal token autoregressively.

---

##### Modules

```text
ShallowTransformer(x[1:T]) -> h_L[1:T]

ConceptHead(h_L[t]):
    logits = linear(h_L[t])
    z = gumbel_softmax(logits, hard=True)
    e = e_hard + (e_soft - stop_grad(e_soft))
    if argmax(z) == <NULL>: return None
    else: return concept_token
```

```text
MiddleTransformer(concept_tokens) -> concept_tokens_updated
# ONLY concept tokens go through these layers
```

```text
DeepTransformer([concept_tokens_updated ; tail_hiddens]) -> logits
```

---

##### Boundary / Tail Definition

```text
τ(t) = index of most recent normal token whose concept != <NULL>
tail(t) = h_L[τ(t)+1 : t]    # contiguous, variable length
```

---

##### Training Forward (single sequence)

```text
h_L = ShallowTransformer(x[1:T])

concept_tokens = []
compressed_pos = set()

for t in 1..T:
    c = ConceptHead(h_L[t])
    if c != None:
        concept_tokens.append(c)
        compressed_pos.add(t)

concept_tokens_mid = MiddleTransformer(concept_tokens)

for t in 1..T-1:
    τ = max{i <= t | i in compressed_pos} or 0
    tail = h_L[τ+1 : t]

    Z_t = concat(concept_tokens_mid_prefix(t), tail)
    logits_t = DeepTransformer(Z_t)
    loss += CE(logits_t, x[t+1])
```

---

##### Attention Rules (DeepTransformer)

```text
Concept -> Concept : allowed (causal or full)
Tail    -> Concept : allowed
Tail    -> Tail    : causal
Concept -> Tail    : disallowed
```

---

##### Key Constraints

* Middle blocks never see normal tokens.
* `<NULL>` means “this token is NOT represented in deep context”.
* Concept tokens must replace information of compressed normal tokens.
* End-to-end backprop through:
  shallow → concept head (ST) → middle → deep.

---

如果你下一步要**真正写 PyTorch 代码**，我可以继续帮你做三件事之一（你选一个就行）：

1. 把上面伪代码 **翻成最小 PyTorch 类结构**
2. 专门写 **attention mask / padding 逻辑**（这是最容易写错的）
3. 设计一个 **O(T) 而不是 O(T²)** 的 batching / forward 实现方式


## 究竟怎样变长？
1. 用整个LLM 纯自回归 
2. NULL tokens
3. 加和 + 阈值
4. 固定K -> 每N个普通tokens生成M个latent tokens
5. 先MTP生成若干普通tokens，maybe再用一些启发式方法终止（比如高熵的tokens）

## v3: pure AR
* 优势：
  * 参数更少
  * 内存占用更少
  * 更快（不需要prefill多次）

* how to make it more non-trivial?
  * 1. 2d-RoPE × （语义根本就不对）--> 都从1开始计数，不同种类的tokens加一个固定bias
  * 2. 多层次concept (可结合2d-RoPE)
  * 3. 从目标文本 X 自动抽取可监督信号 𝑦(𝑋)，让 Planner 必须能从 C 预测这些 y

----

好，下面是**按你最新修改**后的**简洁中英双语 pipeline 总结**，可以直接作为给 codex 的提示词使用（已满足：不同 concept 类型→不同词表；统一 RoPE；不提 segment_pos）。

---
### prompt
#### 中文（简洁提示词）

目标：Planner–Executor 两阶段 **Concept-first Decoding**。concept tokens 是内部高层计划，Executor 仅依赖 concept 生成普通文本。

**Pipeline：**

1. **Planner（概念规划）**

   * 输入：用户指令/上下文 (u)
   * 输出：多种类 concept tokens，每一类来自**独立词表**：
     [
     C = {C^{(1)}, C^{(2)}, \dots}, \quad C^{(i)} \subset V_c^{(i)}
     ]
   * 每一类 concept 为变长序列，以 `<EOS>` 结束。

2. **构造 Executor 输入（仅 concept）**

   * 将不同种类的 concept 串联，中间用对应的分隔符：
     `[<BOS>, <SEP_1>, C^{(1)}, <SEP_2>, C^{(2)}, ...]`
   * **位置编码**：

     * 每种 concept token 的 position index **各自从 1 开始计数**
     * 所有 token **统一使用 RoPE**
   * **Type embedding**：

     * 为每种 concept 类型（以及对应的 `<SEP>`）加入 type embedding
   * 最终输入 embedding：
     `tok_emb + RoPE(pos) + type_emb(type_id)`

3. **Executor（展开生成）**

   * 条件生成：
     [
     x_{1:T} \sim p(x \mid C)
     ]
   * Executor **不读取普通 tokens 前缀作为条件上下文**
   * 普通 token 按标准自回归生成，支持 KV cache。

---

#### English (concise prompt)

Goal: Two-stage **Planner–Executor Concept-first Decoding**. Concept tokens are internal high-level plans; the Executor generates text conditioned only on concepts.

**Pipeline:**

1. **Planner (Concept planning)**

   * Input: user instruction/context (u)
   * Output: multiple **typed concept sequences**, each from a **separate vocabulary**:
     [
     C = {C^{(1)}, C^{(2)}, \dots}, \quad C^{(i)} \subset V_c^{(i)}
     ]
   * Each concept sequence is variable-length and ends with `<EOS>`.

2. **Build Executor input (concept-only)**

   * Concatenate concept sequences with type-specific separators:
     `[<BOS>, <SEP_1>, C^(1), <SEP_2>, C^(2), ...]`
   * **Positional encoding**:

     * Position indices **restart from 1 for each concept type**
     * **Unified RoPE** is applied to all tokens
   * **Type embedding**:

     * Add a learned type embedding for each concept type (and its separator)
   * Final embedding:
     `tok_emb + RoPE(pos) + type_emb(type_id)`

3. **Executor (Expansion / generation)**

   * Generate text as:
     [
     x_{1:T} \sim p(x \mid C)
     ]
   * The Executor conditions **only on concept tokens** (no normal-token prefix as input).
   * Standard autoregressive decoding with KV cache.

## v3.1: speed up training
1. 预测concept tokens时MTP （有点类似于AR结合query）
2. 预测不同type的concept tokens时并行跑
3. z每隔M步更新一次
4. 先用短文本训稳，后续再用长文本

## 你可以做什么实验
1. 随机去掉一些concept tokens，看看重建效果是否会受很大影响，从而证实executor是真的在用concept tokens

下面这些指标都**很便宜**（基本来自已有 forward 的 logits / loss / beam 结果），但能很好地判断：E-step 搜索有没有跑偏、planner 有没有学到东西、以及你这个 ST（(e_{hard}+e_{soft}-\text{stopgrad}(e_{soft}))）有没有在“骗梯度”。

我按模块给一套最实用的监控清单，并标注“正常/异常”信号。

---
----
----
----
## A. Executor 是否真的在用 z（最关键）

### 1) 条件互信息代理：ΔNLL（ablation 差）

每隔一段（比如每 500～2000 step），在一个很小的 mini-batch 上做两次 teacher forcing：

* 正常：用当前选的 (z^*) 计算 ( \text{NLL}(y\mid z^*))
* 破坏 z：把 z 换成

  * 同 batch 里别的样本的 z（shuffle z），或
  * 全 <MASK>/<PAD>，或
  * 随机 z（同长度）

记：
[
\Delta = \text{NLL}(y\mid z_{\text{bad}})-\text{NLL}(y\mid z^*)
]

**正常**：Δ > 0 且逐步变大/稳定（说明 z 有信息）
**异常**：Δ≈0（executor 忽略 z；posterior collapse 风险）

> 这不是“大量计算”：只是在小 batch 上多跑一次 forward。

---

## B. E-step 搜索质量与稳定性（Hard-EM 是否健康）

### 2) Top-1 margin（胜出幅度）

你在 E-step 会得到 top-k 候选的评分 (S(z)=\log p_\theta(y|z)+\log p_\phi(z|u))。

监控：
[
m = S(z^{(1)})-S(z^{(2)})
]

**正常**：m 适中（有区分度，但不总是巨大）
**异常1**：m≈0 长期（搜索不分胜负，噪声大，z* 不稳定）
**异常2**：m 巨大且很早出现（过早塌缩到固定模式/低多样性）

### 3) z* 变更率（assignment flip rate）

每次你更新 E-step 时，统计：

* 有多少样本的 (z^*) 跟上次不同（或编辑距离>阈值）

**正常**：前期高、后期下降
**异常**：一直很高（模型在震荡/学习率过大/评分不可靠）
**异常**：很早降到接近 0（过早锁死，可能陷入局部最优）

> 这个几乎零成本：你本来就有 z*，只要做个 hash/编辑距离。

### 4) “旧 z* 仍好不好用”（staleness gap）

在稀疏更新 E-step 时：
同一个样本，用旧 (z^**{\text{old}}) 和新 (z^**{\text{new}}) 的分数差：

[
g = S(z^**{\text{new}})-S(z^**{\text{old}})
]

**正常**：g 小且逐步变小（说明稀疏更新没伤太多）
**异常**：g 常常很大（说明你 stale 太久，应该更频繁更新或减小LR）

---

## C. Planner 是否在学（以及 ST 是否“对齐”）

### 5) Planner 熵（或 top-1 概率均值）

从 planner 的 soft 分布里取一个便宜统计：

* 每个位置的 entropy (H(p_t)) 的均值
* 或 mean max-prob：(\mathbb{E}[\max_i p_t(i)])

**正常**：前期高熵→中期逐步变尖→后期稳定
**异常1**：一直高熵（planner 没学到可用 latent）
**异常2**：很快极低熵（过早确定、候选多样性不足、Hard-EM 易卡死）

### 6) e_soft 与 e_hard 的一致性（cosine / L2）

你本来就算了 (e_{soft}) 和 (e_{hard})。监控：

* (\cos(e_{soft}, e_{hard})) 的均值（按 token 或按序列平均）
* 或 (|e_{soft}-e_{hard}|)

**正常**：随训练逐步更一致（soft 更接近 one-hot）
**异常**：长期不一致（ST 梯度在优化“模糊 embedding”，forward 却走硬 token，容易 drift）

> 这个也几乎零成本，因为 embedding 已经在 forward 里了。

### 7) Planner 梯度范数（全局或最后一层）

记录一个简单标量：

* (|\nabla_\phi|)（global grad norm）
* 或某层 grad norm

**正常**：有信号、不过度爆炸
**异常**：接近 0（executor 不依赖 z / ST 无有效信号）
**异常**：频繁爆炸（温度过低、学习率过大、评分震荡）

---

## D. 输出侧：训练是否在“长度扩展”时崩

### 8) 分段 NLL（按位置桶）

不需要额外 forward：你 teacher forcing 本来就算了每个 token 的 loss。
把 loss 按位置分桶（比如 0-64, 64-128, …）求均值。

**正常**：各桶都逐步下降；扩展长度时后段先抖再稳
**异常**：只前 64 降，后面一直不动（z 可能只服务开头 / executor 长程没学到）

---

# 一套最小但够用的监控组合（建议你就用这 6 个）

1. **ΔNLL(z-shuffle)**：executor 是否用 z
2. **Top-1 margin**：E-step 区分度
3. **z* flip rate**：稳定性/是否震荡
4. **staleness gap**：稀疏更新是否过头
5. **planner entropy / max-prob**：是否塌缩或没学
6. **cos(e_soft, e_hard)**：ST 对齐是否健康

这些几乎都不需要额外大计算，只有第1个需要偶尔多一次 forward（小 batch 即可）。

---

## 快速“异常→动作”对照表

* ΔNLL≈0：加强 bottleneck（executor 不看 u）、增大 z 容量/词表、降低 executor 学习率、提高 z dropout/扰动对比
* margin≈0 且 flip 高：提高候选 K 或多样性；E-step 更频繁；降低学习率
* flip 很早变 0 + 熵很低：提高温度/熵正则；用 diverse beam；偶尔强制探索（采样候选）
* staleness gap 大：缩短 E-step 更新间隔或用自适应更新；减 LR
* cos(e_soft,e_hard) 长期低：降温度或 anneal；对 soft 加 sharpen（但别一下子太硬）
