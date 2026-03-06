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
     x_t ~ p(x_t​∣C,x_{ < t} ​)
   * Executor **不读取普通 tokens 输入前缀作为条件上下文**
   * 普通 token 按标准自回归生成，支持 KV cache。
   * 会把已生成普通 tokens 作为“条件上下文”再输入模型

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

### 上下文
#### A
我提出了一个LLM推理加速的方案：先用LLM将输入自回归压缩到少量的离散concept tokens组成的序列中（来自一个新的词表），再用同一个LLM将concept tokens序列自回归生成输出。简要总结我这个方法，确保你理解了我的方案的细节，后续对话将在此基础上进行。一些更具体的细节如下: ……（将前面的中文prompt粘贴上来）
#### B
通过分析你提供的代码（`model.py`、`train.py`、`infer.py` 等），我为你总结了这个项目的模型架构与训练流程。这是一个**基于“概念规划（Planner）- 执行（Executor）”两阶段生成框架**的 Token 压缩/受控生成模型。

##### 1. 核心模型架构 (`SharedBackboneUnifiedHead`)

该模型旨在通过**共享骨干网络**（Shared Backbone）同时完成“概念规划”和“文本生成”两个任务。

*   **基础骨干 (Backbone)**:
    *   使用一个预训练的因果语言模型（`AutoModelForCausalLM`）作为特征提取器。
    *   集成了 **LoRA (Low-Rank Adaptation)** 进行参数高效微调，大部分基础参数被冻结，只训练 LoRA 权重。
*   **统一嵌入层 (Unified Embeddings)**:
    *   **Token Embedding**: 分为 `base`（原模型词表）和 `new`（新增的概念 Token）。
    *   **Type Embedding**: 引入了 `type_embed`，用于区分当前是在处理普通文本、还是某种特定类型的概念（如 `ConceptTypeConfig` 定义的多种类型）。
    *   输入最终是 `Token Embedding + Type Embedding`。
*   **统一输出头 (Unified Output Head)**:
    *   **Base Head**: 映射回原词表大小，用于生成普通文本。
    *   **New Head**: 映射到新增的概念词表，用于生成概念 Token。
    *   前向传播时，根据当前任务输出拼接后的 Logits。
*   **配额控制器 (PlannerQuotaController)**:
    *   用于在规划阶段动态调节生成的“配额”（Quota），控制生成数量或分布。

##### 2. 训练流程 (两阶段训练)

训练在一个 Step 中分两个阶段进行，两个阶段共享梯度更新。

###### 第一阶段：规划器 (Planner Stage)
**目标**：根据输入文本，生成一系列高层的“概念”Token。这是一个**自回归生成的**过程，但在训练时是在线生成的。

1.  **构建 Prompt**: `[BOS, input_ids, <PLAN>]`。模型看到 `<PLAN>` 后开始生成概念。
2.  **异步概念生成 (`plan_concepts`)**:
    *   模型并行地为定义好的多种 `ConceptType` 生成序列。
    *   **机制**：使用 Gumbel-Softmax 进行**可微采样**（Soft/Hard 结合），使得后续的执行器可以通过梯度回传优化规划器。
    *   **约束**：
        *   `commitment_loss`: 约束软采样（Gumbel Softmax 输出）和硬采样（One-hot）的一致性。
        *   `usage_kl_to_uniform`: 防止概念坍缩（Concept Collapse），鼓励多样性。
        *   `loss_eos`: 鼓励模型适时生成 EOS，控制概念序列长度。
        *   `loss_quota`: 通过配额控制器约束生成的总量。
3.  **输出**: 得到每个类型的概念序列（Hard ID 用于构建 Prompt，Soft Embeddings 用于可能的梯度流或分析）。

###### 第二阶段：执行器 (Executor / BPTT Stage)
**目标**：基于规划器生成的概念序列作为前缀（Prefix），重建/生成原始文本。

1.  **构建执行前缀 (`build_executor_prefix`)**:
    *   将第一阶段生成的概念序列（包括 Type ID、Position ID 等）拼接起来。
    *   结构：`[Concept_Type_1..., Concept_Type_2..., BOS]`。
2.  **前向传播与计算 Loss**:
    *   **输入**: `[Executor Prefix, BOS, input_ids]`
    *   **目标**: 预测 `input_ids`（Next Token Prediction）。
    *   **Mask**: 执行器通过 Attention Mask 可以看到所有的概念前缀，但概念前缀内部是基于 Causal Mask 或特定可见性设置的。
    *   **Loss**: 标准的 CrossEntropyLoss，衡量在给定概念规划下恢复原文本的能力。

##### 3. 数据流总结

1.  **Input Text** -> **Backbone** -> **<PLAN>** -> **Planner Generation (Gumbel Softmax)** -> **Concept Tokens**.
2.  **Concept Tokens (作为 Prefix)** + **Input Text** -> **Backbone** -> **Text Reconstruction Loss**.

##### 关键点总结
*   **端到端可微**：通过 Gumbel-Softmax 或类似的直通估计（Straight-Through Estimator），让执行器的 Loss 可以更新规划器的参数，使模型学会生成“有利于还原文本”的概念。
*   **多类型概念**：支持多种不同属性的概念（由 `metas` 定义），模型会按顺序或并行规划这些不同层级的压缩表示。
*   **LoRA 微调**：保证了训练效率，主要调整的是 LoRA 参数和新增的 Head/Embedding 参数。

这个架构非常适合**文本压缩**、**受控生成**或**长文本建模**（通过高层概念减少序列长度）任务。我已完全理解该架构。

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



-----


# position_ids设计

# bugs:
* padding tokens 也计入position_ids的递增
* concept tokens貌似没有PAD，只有EOS


# debug
1. 不用base tokens
2. 究竟能不能看到输入

# debug结果
我进行了以下修改：
1. 稍微调整了辅助损失的占比，放大了loss_unif的占比
2. 减少concept tokens词表的词数量以及种类
3. 稍微改进了pos_id的使用逻辑
现在似乎又一点进展如下：concept tokens序列多样性稍微好了一些，同一个step的eval中不同样本的concept tokens展现了一定差异性
但是仍然有问题：同一个step的eval中不同样本的concept tokens其实还是比较接近，而且容易出现连续几个concept tokens都相同的情况，最终output重建结果也很差
这可能是因为训练还没够吗（因为我只训了不到1 epoch）

# 进一步改进
1. 增加一些调试信息，记录训练时concept tokens的使用情况
2. input corruption / window / noisy embedding
3. 利用concept tokens 预测文本的一些全局属性
4. 不对称lr

# 训练时gumbel + greedy混合，noisy embedding, 且降低tau_min (b99778c0a93f6e653c33ab44fdf843d55aebbe60)
## 变好：
1. concept序列基本不会重复
2. 同一个concept序列中不太会出现重复的concept tokens
## 仍然不好：
1. 重建效果还是很糟，几乎没有还原原意
2. 还是有一点点同一个concept序列中会出现个别重复的concept tokens
## 其他观察
1. loss_unif从之前的1+降到0.25-
2. loss_rec可以降低到3左右
3. loss_eos降到0.03左右
4. loss_commit逐渐走高，到8e-6左右
5. 训练时concept diversity逐渐走低，unique率到大概0.75左右
## 例子（900steps）

797 - INFO - [Eval Step 900] Sample 0
797 - INFO - Input: Iran also launched several retaliatory air raids on Iraq, while primarily shelling border cities such as Basra. Iran also bought some Scud missiles from Libya,
797 - INFO - Concepts:
798 - INFO -   concept: ids=[152446, 151680, 152050, 152656, 152656, 151680, 151680, 151671, 151670] tokens=<C_775> <C_9> <C_379> <C_985> <C_985> <C_9> <C_9> <C_0> <EOS_CONCEPT>
798 - INFO - Output: The 1960s saw the rise of the "new" American cinema, which was characterized by a more realistic and realistic portrayal of American life.
810 - INFO - ====================================================================================================
816 - INFO - [Eval Step 900] Sample 1
817 - INFO - Input: London has played a significant role in the film industry. Major studios within or bordering London include Twickenham, Ealing, Shepperton, Pinewood
818 - INFO - Concepts:
818 - INFO -   concept: ids=[152446, 151680, 152022, 152022, 152351, 152406, 151680, 152598, 151670] tokens=<C_775> <C_9> <C_351> <C_351> <C_680> <C_735> <C_9> <C_927> <EOS_CONCEPT>
819 - INFO - Output: The first major change in the game was the introduction of the "Bingo" mode, which was introduced in 1999. This mode was a
821 - INFO - ====================================================================================================
821 - INFO - [Eval Step 900] Sample 2
822 - INFO - Input: The industry reached its peak in the 1920s employing around 18,000 people. In modern times however the textile industry declined
823 - INFO - Concepts:
823 - INFO -   concept: ids=[152535, 152637, 151680, 152050, 152662, 152245, 151680, 152598, 151670] tokens=<C_864> <C_966> <C_9> <C_379> <C_991> <C_574> <C_9> <C_927> <EOS_CONCEPT>
824 - INFO - Output: The 2010 census showed that 10.6% of the population was of European descent, 12.6% of the population
826 - INFO - ====================================================================================================
826 - INFO - [Eval Step 900] Sample 3
826 - INFO - Input: Spider-Gwen
Captain America is a S.H.I.E.L.D. agent on Earth-65, who apprehends Spider-Gwen during her battle with
827 - INFO - Concepts:
828 - INFO -   concept: ids=[152065, 152155, 152308, 152308, 152624, 152624, 152624, 151671, 151670] tokens=<C_394> <C_484> <C_637> <C_637> <C_953> <C_953> <C_953> <C_0> <EOS_CONCEPT>
828 - INFO - Output: History
Early history
The earliest known reference to the city of Alexandria is found in the 1st century BC, in the writings of the Egyptian historian P

