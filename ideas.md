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
