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


