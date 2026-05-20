# MoE 基础概念与因果链 — 入门教学版

> **写给谁**：手上有这套笔记、但还没把 MoE 的因果关系想清楚的你。
> **怎么读**：从第 1 节按顺序读到第 12 节；每节末尾的"一句话"是你应该形成的 mental model。读完不需要记住所有公式，但要能用自己的话讲出"为什么 X 这样设计"。
> **不会讲什么**：具体超参怎么调（看 22_FINAL）、各 paper 的精确实验数（看 01-30）。

---

## 0. 阅读路径

```
1. 为什么发明 MoE       ← 一切的起点
        ↓
2. Active vs Total      ← MoE 唯一不能不知道的两个数
        ↓
3. 路由 (Routing)       ← MoE 的灵魂与最难的部分
        ↓
4. Load Balance         ← 路由必然伴生的麻烦
        ↓
5. Fine-grained 革命    ← 2024 年的关键突破
        ↓
6. Shared Expert        ← 国产派 vs 西方派的分水岭
        ↓
7. Attention 字母汤     ← MHA / GQA / MLA / Mamba 是什么、为什么
        ↓
8. MTP (Multi-Token Prediction)
        ↓
9. Wind Tunnel          ← 小模型预演大模型的方法论
        ↓
10. 工程语汇 (EP / TP / DP / Dropless / Megablocks)
        ↓
11. MoE 特有的不稳定问题与解药
        ↓
12. 把这些回到 22_FINAL spec — 每个决策的因果链
```

---

## 1. 为什么发明 MoE

### 问题的起点

模型越大越聪明 —— 这是 OpenAI 2020 年 Kaplan / Hoffmann (Chinchilla) 等 scaling law 论文的实证结论。**参数越多，loss 越低，下游能力越强**。

但有个麻烦：
- **Dense 模型每生成一个 token，所有参数都要被算一遍**
- 100B 参数的 dense 模型 → 推理时每 token 算 100B FLOPs
- 1T 参数的 dense 模型 → 推理时每 token 算 1T FLOPs
- 推理成本 ∝ 参数量 → 大模型部署贵

### MoE 的核心想法（一句话）

> **把参数变多，但每次只用一小部分**。

类比：
- Dense 模型 = 一个全能的天才，每次回答都用上全部知识
- MoE 模型 = 一群专家委员会，每次只叫 K 个相关的专家来回答

具体来说，把 dense 模型的 FFN 层（前馈层，占 ~70% 参数）换成 N 个并行的"小 FFN"（叫 expert），然后有个 router 决定每个 token 该送到哪 K 个 expert。

### 为什么这样有用

| | Dense 16B | MoE 16B/2.4B-active |
|---|---|---|
| 总参数 | 16 B | 16 B |
| 推理 FLOPs/token | 16 B | **2.4 B** (省 7×) |
| 训练 FLOPs/token | 16 B | ~3 B (省 5×；梯度只算 active expert) |
| 部署知识容量 | 16 B | **16 B** (一样) |
| 单卡 GPU 显存占用 | ~32 GB | ~32 GB（全 expert 都要驻留） |

**关键**：MoE 的便宜是**算力**便宜，不是**显存**便宜。这一点很多人混淆。N 个 expert 都要驻留在 GPU 显存里，只是计算时不全部参与。

### 为什么 2024 之后才火

MoE 概念 1991 年就有（Jacobs et al.）；Google 2017 发表 Sparsely-Gated MoE；2021 Switch Transformer。但**直到 2024 年才真正成为 LLM 主流**，原因：

1. **DeepSeekMoE (2024-01)** 证明了"很多小 expert 比少数大 expert 好"——这是第 5 节会讲的 fine-grained 革命
2. **Mixtral 8x7B (2023-12)** 是第一个性能强势的开源 MoE，把社区拉过来
3. **Aux-Loss-Free balancing (2024-08)** 解决了 router 训练不稳的老大难

### 一句话总结

> **MoE = 把 FFN 换成"许多 expert + 路由器"，参数变多但每次只用 K 个 expert，因此推理成本只与 active 参数相关，而非总参数。**

---

## 2. Active vs Total Params — MoE 唯一不能不知道的两个数

### 定义

| 名词 | 含义 | 决定什么 |
|---|---|---|
| **Total params (总参数)** | 模型所有参数（包括所有 expert + attention + embedding） | **知识容量**、显存占用 |
| **Active params (激活参数)** | 每个 token 推理时实际经过的参数（仅 K 个 selected expert + attention + embedding） | **每 token FLOPs**、推理 latency、训练算力 |
| **Sparsity (稀疏度)** | Active / Total 比例 | MoE 设计的"取向" |

### 经典几个例子（每个都记一下，会有直觉）

| 模型 | Total | Active | Sparsity | 含义 |
|---|---|---|---|---|
| DeepSeek-V3 | 671 B | 37 B | 1/18 | 极大模型，超稀疏 |
| Kimi K2 | 1 T | 32 B | 1/31 | 更稀疏 |
| Mixtral 8x7B | 46.7 B | 12.9 B | 1/3.6 | 早期，稠密 |
| Llama 4 Maverick | 400 B | 17 B | 1/23 | 极稀疏 |
| 你的 16B Profile B | 15.5 B | 2.4 B | 1/6.5 | 中等 |

### 因果链

- **Active 大** → 推理 FLOPs/token 多 → latency 高，部署贵，但 quality 通常更好
- **Total 大** → 知识容量大 → 下游 task 上限更高，但需要更大显存
- **Sparsity 高（active/total 小）** → 同 active 算力下能"塞"更多参数 → 适合大量知识、长尾问题

类比：**Active 是"现场叫来开会的专家数"，Total 是"花名册总人数"**。Sparsity 是"出席率"。

### 两套口径（你的笔记一直在 hammered 这点）

**严格口径**：只算 attention + 选中 expert FFN + norm
**V3 口径**：上面 + embedding + LM head

> V3 把 embedding 算进 active 是 marketing 上有用的 — 让 V3 "37B active" 听起来更大；但严格说，embedding 是 lookup 不是 compute，把它算进 FLOPs 计算是不准的。

| 你的 16B Profile B | 严格 | V3 口径 |
|---|---|---|
| Active | 2.4 B | 2.9 B |

**和别人对比时一定要问"用哪个口径"** —— Ling-mini-2.0 说"1.4 B active" 用 V3 口径，严格只有 ~1.1 B。

### 一句话总结

> **Total 决定知识容量与显存，Active 决定推理算力与训练算力；MoE 的核心 trick 就是把这两个数解耦开。**

---

## 3. 路由 (Routing) — MoE 的灵魂

### 什么是路由

每个 token 进入 MoE 层时，需要决定"我去找哪 K 个 expert"。这个决定由一个叫 **router** 的小模块完成 —— 它通常就是一个线性层 (`x @ W_router`)。

```
token x (维度 D)
   ↓
W_router (D → N)
   ↓
N 个分数 (logits, 每个对应一个 expert)
   ↓
Top-K 选择 → 选出 K 个分数最高的 expert
   ↓
weighted sum(expert_i(x) for i in top_K)
```

### Top-K 是什么、为什么

**Top-K** = 选 K 个分数最高的 expert，让 token 经过这 K 个 expert，输出按 router 给的权重加权求和。

| K | 谁用 | 因果 |
|---|---|---|
| K=1 | Switch (2021), Llama 4 (2025) | 极简、最便宜；但 quality 通常输给 K≥2，因为单一 expert 表达力有限 |
| K=2 | Mixtral, DBRX, Grok-1 | 2024 主流；平衡 |
| K=6-8 | DeepSeek-V2/V3, Ling 2.0, Qwen3 | 2025 主流；更细的 expert 组合 → quality 更高 |

**为什么 K 越大越好但不能无限大**：
- K → ∞：变成 dense FFN，没有节省了
- K=1：组合空间太小
- K=8 在 N=64 时组合空间 C(64,8) ≈ 4.4 亿 → 充分多样性

### Router 怎么学

这是 MoE 最微妙的部分。**Router 没有标签**——你不知道哪个 expert 该接哪个 token，只能通过梯度自己摸索：

1. 初始时 router 随机 → 把 token 随便发给 expert
2. 每个 expert 输出 → 整体 loss
3. 反向传播 → router 的 W 收到梯度
4. Router 慢慢学会"这种 token 送 expert A 效果好"

### 但这里有个鸡生蛋的问题

- 一开始 router 随机选 → 偶然 expert\_5 被多选了
- Expert\_5 拿到更多数据 → 训得更好
- Router 发现 expert\_5 总是给好答案 → 更多 token 路由给它
- 雪球效应 → 几个 expert 包揽所有 token，其他 expert **饿死 (dead experts)**

这就是为什么 MoE 必须解决 **load balance** 问题（第 4 节）。

### 类比

Router 就像门卫派单：
- 门卫看到一个人（token），决定送去哪 K 间办公室（expert）的
- 一开始门卫凭感觉派 → 某几间办公室人特别多
- 那些办公室处理得越来越熟练 → 门卫更频繁送过去
- 其他办公室没活干 → 员工生疏 → 越没人送 → 恶性循环

**Router 是 MoE 模型里最受过分宠爱、又最容易翻车的小模块**。

### 一句话总结

> **路由 = router 决定每个 token 送哪 K 个 expert；router 是 MoE 的灵魂，但因为它没有直接标签且容易雪球退化（dead experts），所有 MoE 设计的核心难点都围绕"如何让 router 收敛得既准确又均衡"。**

---

## 4. Load Balance — 路由必然伴生的难题

### 为什么"负载均衡"是 MoE 的核心问题

承接第 3 节的雪球：如果不干预，N 个 expert 中只有 5-10 个会被持续使用，其他几十个是 "dead experts"。这有 3 个糟糕后果：

1. **浪费参数**：训了几十亿参数，但只有 1/10 真正被用
2. **训练不稳**：被宠爱的 expert 拿到的 batch 太大，梯度方差也大，loss 容易 spike
3. **推理瓶颈**：在 EP（expert parallel，第 10 节）下，token 集中流向几张卡 → 通信带宽和计算时间被那几张卡卡住

所以 MoE 一定要"强制" router 把 token 比较均匀地分到 N 个 expert。

### 解药 1：Aux Loss（辅助损失，Switch / Mixtral 派）

加一个额外的损失项，专门惩罚负载不均：

$$L_{aux} = \alpha \cdot N \cdot \sum_i f_i \cdot P_i$$

- $f_i$ = expert i 实际拿到的 token 比例
- $P_i$ = router 对 expert i 的平均概率
- $\alpha \approx 0.01$

直觉：如果某个 expert 既"实际拿很多"又"router 倾向给它"，就重重罚一笔。Router 被迫学会均匀分配。

**问题**：这个 loss **同时塑造 router**（梯度直接打在 W\_router 上）。它把 router 从"质量最优"拉向"负载均衡"。**质量与均衡是两个互相拉扯的目标**，α 调大 → 太均衡但 quality 差，α 调小 → quality 好但 dead experts 多。

### 解药 2：Aux-Loss-Free / ALF（V3 派，2024-08 突破）

引入一个**纯 controller 变量 bias**：

- Router 算出 logits → 加一个 bias → 再选 top-K
- **bias 不进梯度**（不被 optimizer 学习）
- 每 step 之后单独更新：哪个 expert 负载多 → bias 减；负载少 → bias 加

```python
# 每步训完后
load = 各 expert 实际收到的 token 比例
bias -= γ * sign(load - 1/N)  # γ ≈ 0.001
```

**精妙之处**：bias 只影响"谁被选"，不影响"被选后给多大权重"。Router 的 W 仍然只接受 quality loss 的梯度。**质量与均衡解耦了**。

类比：
- Aux loss = 一边教员工干活一边打他："分配不均你也要负责" → 员工迷茫
- ALF = 设一个独立的 HR：员工照常干活，HR 看到谁忙就少派单 → 员工专心干活

### 演化时间线

- 2017-2023：Switch / GShard / Mixtral 都用 aux loss
- 2024-08：Wang et al. 论文 (arxiv 2408.15664) 提出 ALF
- 2024-12：DeepSeek-V3 首个大规模用 ALF（671B / 14.8T tokens）
- 2025+：Kimi K2 / Moonlight / Ling 2.0 / GLM-4.5 / Ring 跟进 → 1/3 新模型都用 ALF

但 Mixtral 派（aux loss）也没死 —— OLMoE / Qwen3 / gpt-oss / Hunyuan / Llama 4 仍用。两条路线现在各占半壁江山，**没有 paper 直接 head-to-head 比较过**，这是你 wind tunnel A2 T2.1 要回答的问题。

### 一句话总结

> **Load balance 是 MoE 必须解决的问题，否则雪球退化为 dead experts；解药两条路线：aux loss（同时塑造 router，2024 主流）vs ALF bias（独立 controller，2024-08 起新主流），目前各占半壁江山。**

参考：你的 `papers/03_auxloss_free.md`（ALF 原始论文）+ `30_routing_implementation.md`（两条 kernel 的实现差异）

---

## 5. Fine-grained 革命 — 2024 年的关键突破

### 旧范式（Mixtral 2023）

- N = 8 个 expert，每个 expert 的 FFN 中间维度 = 14336
- Top-K = 2
- 每个 token 经过 2 个 "大块" expert

### 新范式（DeepSeekMoE 2024-01）

- N = 64 个 expert，每个 expert 的 FFN 中间维度 = 1408 (只有 1/10 大)
- Top-K = 6
- 每个 token 经过 6 个 "小块" expert

### 为什么 fine-grained 更好

**总 FFN 参数差不多**（64 × 1408 ≈ 8 × 14336）—— 但**组合数量**天差地别：

- Mixtral: C(8, 2) = **28** 种 expert 组合
- DeepSeekMoE: C(64, 6) = **74 千万** 种组合

类比：
- Mixtral = 8 个大学科系，每次选 2 个 → 28 种"学科组合"
- DeepSeekMoE = 64 门小课，每次选 6 门 → 7400 万种"课表"

更多组合 → 模型可以为不同类型的 token 学到更精细的功能划分 → 同 active params 下 loss 更低（DeepSeekMoE 实验 ~0.02-0.05 loss 改善）。

### 后续演化

- 2024-05 DeepSeek-V2：N=160, K=6
- 2024-12 DeepSeek-V3：N=256, K=8
- 2025-09 Ling-mini-2.0：N=256, K=8（在 16B 总参上）
- 2025-09 Qwen3-Next：N=512, K=10 （极致）

**N 越多越好，但不是无上限**：N=256 时 EP all-to-all 通信开销变大，kernel (Megablocks) 复杂度上升。N=64-128 是当前甜区。

### 一句话总结

> **Fine-grained = 拆成更多、更小的 expert，组合空间指数级增长，同 active 算力下质量提升；2024-01 DeepSeekMoE 验证后成为业界标准，N=64-128 是 2025+ 主流甜区。**

参考：`papers/01_deepseekmoe.md` + `17_finegrained_scaling.md` (Krajewski et al. scaling law)

---

## 6. Shared Expert — 国产派 vs 西方派的分水岭

### 什么是 shared expert

除了 N 个 routed expert（用 top-K 选）外，还有 1-2 个 **shared expert**，**每个 token 永远都经过它们**（不参与 top-K 选择）。

```
token x
   ↓
   ├── shared expert(s) (永远经过)  ──┐
   │                                  │
   └── routed expert × K  ────────────┼── 求和 → 输出
                                      ┘
```

### 为什么这样做

直觉：很多 token-level 操作（语法、空格、标点、词形变化）是**所有 token 都需要的通用功能**。如果让 routed expert 都学这些 → 重复浪费；让 shared expert 包办 → routed expert 可以专注真正"专家级"的知识。

类比：大学里 shared expert 是"必修课"（人人都上），routed expert 是"选修课"（按兴趣选）。

### 国产派 vs 西方派

| 派系 | 用 shared | 代表 |
|---|---|---|
| **国产派 / DeepSeek 系** | ✓ 1-2 个 | DeepSeek-V2/V3, Kimi K2, Ling 2.0, Hunyuan, Qwen2-MoE, GLM-4.5, Step-3 |
| **西方派 / 西方风格** | ✗ 0 个 | Mixtral, DBRX, OLMoE, Grok-1, gpt-oss, Phi-3.5 |
| **半派** | 半个？ | Llama 4 (用 shared MLP 但不叫 shared expert), Snowflake Arctic (dense residual) |

### 因果是开放问题

**这是公开文献里最大的分歧**。两边都有 SOTA：
- 国产派论据：V3 在 14.8T 训练上验证；shared expert 让 routed expert 更"纯粹"
- 西方派论据：OLMoE 论文 §3 实验显示 0 shared 在 1B/7B 上更好；多一个 shared expert 增加部署复杂度

**没有任何论文做过严格的 head-to-head 对比** —— 这是你 wind tunnel A2 T2.4 要回答的问题。

### 一句话总结

> **Shared expert = 永远参与的"通用 expert"，让 routed expert 专注 specialization；国产派和西方派各占一半，causally 谁更好仍是公开问题。**

参考：`papers/01_deepseekmoe.md`（首次引入 shared）+ `09_olmoe.md` (0 shared 路线)

---

## 7. Attention 的字母汤 — MHA / GQA / MLA / Mamba 都是什么

这部分技术上**与 MoE 正交**（attention 决定 token 之间怎么交互；MoE 决定 FFN 怎么算）。但每个 MoE 模型都会做 attention 选型，所以需要懂。

### 因果起点：KV cache 的内存压力

Attention 的工作机制：每个 token 生成 Q, K, V 三个向量。生成第 t 个 token 时，要看前面所有 token 的 K 和 V —— 所以 K, V 要存下来（KV cache）。

| 模型 | 生成 1000 token 时 KV cache 占用 |
|---|---|
| GPT-3 175B (MHA) | ~7 GB |
| 现代 70B dense (GQA) | ~1 GB |
| V3 671B (MLA) | ~0.07 GB |

KV cache 太大 → 显存吃光 → batch size 上不去 → 部署贵。**所有 attention 变种都是在解决 KV cache 太大这个问题**。

### MHA (Multi-Head Attention) — 原版

每个 head 独立的 Q, K, V，一共 H 套。简单、稳定，但 KV cache 大。

### GQA (Grouped Query Attention) — 主流 2023+

把 H 个 Q heads 分组共享 K, V。例如 32 Q heads + 4 KV heads → KV cache 缩到 1/8。**牺牲一点点 quality 换大幅 KV cache 缩减**。Llama 2 起几乎所有 dense LLM 都用。

### MLA (Multi-head Latent Attention) — DeepSeek 发明 2024

K, V **先投影到低维 latent (~512 维)** 再展开 —— KV cache 只存 latent。DeepSeek-V2/V3/V3.1/V3.2/K2/LongCat 用。

- 优点：KV cache 比 GQA 还小 10×；context 长度可扩到 100K+
- 缺点：实现复杂、kernel 不易优化、训练 ops 难度高
- 16B 模型上 ROI 不足（KV cache 已经不是主要瓶颈）

### Mamba / Lightning / DeltaNet / SSM — 线性注意力家族

把 softmax attention（O(N²) memory）换成 RNN-like 或线性注意力（O(N) memory）→ 超长 context 也不爆显存。

- 缺点：retrieval 精度不如 softmax
- 主流做法：**混合架构** —— 大部分层用 Mamba/Linear，少数关键层保留 softmax (e.g. Jamba 7 Mamba : 1 Attn, Qwen3-Next 3 DeltaNet : 1 Attn, Granite 4 hybrid)

### 选型因果链

| 你想要什么 | 选什么 |
|---|---|
| 简单稳定、不追求极致 | MHA (老派) |
| 现代标准、KV cache 减半 | **GQA** (16B 默认) |
| 极致 KV cache 压缩 | MLA (但需要 DeepSeek 的 kernel 生态) |
| 1M+ context、长生成 | Hybrid (Mamba/Lightning + Softmax) |

### 一句话总结

> **Attention 变种都是为了减小 KV cache：MHA → GQA (主流) → MLA (DS 系极致) → Mamba/Linear hybrid (超长 context)。与 MoE 正交，但选型互相影响（MLA + MoE 是 V3 标志，GQA + MoE 是 Qwen/Ling 主流）。**

参考：`papers/02_deepseek_v2.md` §3 (MLA)、`05_qwen3.md` (GQA)、`15_jamba.md` (Mamba hybrid)

---

## 8. MTP — Multi-Token Prediction

### 普通 next-token prediction

LLM 训练时每个 token 只预测**下一个 token**。Loss 在每个位置算 cross-entropy。

### MTP 想法（V3 / Gloeckle 2024）

**每个位置预测后面 D 个 token**（D=1, 2, 4...），所有预测的 loss 都加起来。

```
              x_1 → predict x_2  (主任务)
              x_1 → predict x_3  (D=2 时的额外任务)
              x_1 → predict x_4  (D=3 时的额外任务)
                ↓
            训练时把所有 loss 加权求和
```

### 为什么有用

1. **更密的训练信号**：每个 token 不只学"下一个"，还学"未来 D 个" → 学到更长距离的依赖
2. **推理可加速** (self-speculative decoding)：模型已经"预测了未来"，可以一次性输出多个 token 候选，省 latency
3. **零部署成本**（如果只用第 1 个 head）：推理时丢弃额外的 MTP heads，主 head 与不带 MTP 训出来的模型一样用

### 训练代价

V3 风格 D=1：每层加 1 个额外 transformer block → 训练计算 +5-10%
Gloeckle 风格 D=4：并行多 head → 训练 +10-20%

### 哪些模型用

| 用 MTP | 不用 MTP |
|---|---|
| V3, V3.1, V3.2, K2(no)... 等等 | Mixtral, DBRX, Llama 4, gpt-oss, Phi-3.5, OLMoE, Qwen3-30B/235B, Hunyuan-Large |
| GLM-4.5, GLM-4.6 | |
| Ling-mini/flash/1T, Ring 全系 | |
| Qwen3-Next, MiniMax-M2 (D=3!) | |

**有意思的统计**：28_open_source_moe_catalog §3.6 发现，**用 MTP 的模型 70% 用 sigmoid+ALF 路由**；不用 MTP 的模型大多用 softmax+aux。**MTP 和 ALF 几乎是同一个生态圈**。

### 你的 16B 用不用 MTP？

这是 23_mtp_investigation 整篇调研的问题。结论：**2.4B active 处于 boundary case**：
- Gloeckle 2024 数据：≤1.3B active 时 MTP 拖累，≥6.7B 才显效
- V3 用 D=1 在 37B active 上 +0.5-1pt 下游
- 2.4B active 没有直接证据 → wind tunnel A2 T3.1 验证

### 一句话总结

> **MTP = 每位置预测后 D 个 token，提供更密训练信号 + 推理加速可能；训练 +5-10%、推理零成本（丢弃 head）；2.4B active 处于不确定区域，需要 wind tunnel 验证。**

参考：`papers/20_mtp_gloeckle.md` + `23_mtp_investigation.md`

---

## 9. Wind Tunnel — 小模型预演大模型的方法论

### 为什么需要 wind tunnel

训一个 16B / 15T-tokens 的模型，单次试错成本 ~500K H100 hours（≈ 1.5M 美元）。如果架构选错，钱就打水漂。

**Wind tunnel** = 用一系列**小模型（200M → 1B → 4B → 16B 等比缩小版）做架构消融**，先验证设计，再投入主训练。

### 类比

- 真实飞机直接做风洞测试太贵 → 先在缩小版机翼上跑风洞
- 大 LLM 训练直接跑太贵 → 先在缩小版 MoE 上跑 wind tunnel

### Anchor ladder（22_FINAL §8 设计）

| Anchor | 总参 | Active | Tokens | 算力 | 用途 |
|---|---|---|---|---|---|
| A0 | 200M | 30M | 5B | 烧机 | sanity check / kernel 验证 |
| A1 | 5 个 sweep | – | 16B | 拟合 | scaling law 系数 α |
| **A2** | **1B** | **200M** | **25B** | ~580 H100-hr | **关键消融** |
| A3 | 4B | 600M | 80B | ~1200 | scaling 外推 |
| A4 | 16B | 2.4B | 320B | ~3200 | final sanity (1/50 main run) |

A2 是核心 —— 9 个消融 × 27 arms 在这做（详见 29_wind_tunnel_a2.md）：
- T1 训练机制：ε / 优化器 / scheduler
- T2 架构核心：路由派系 / 粒度 / Profile / shared / ALF 变种
- T3 训练辅助：MTP / Block AttnRes

### 为什么 A2 选 1B/200M-active

- 1B 比 16B 小 16× → 训练快 16²-32× （compute 是 N² 增长）
- 200M active 与 16B 的 2.4B 同比例 → 架构 ratio 一致
- 25B tokens 足够看到 loss 收敛、router 健康指标稳定
- 单 arm ~21 H100-hr → 一整套消融 ~580 hours，预算 OK

### 关键 caveats（这是 wind tunnel 的硬限制）

1. **某些架构在小尺度不显效**：MTP 在 200M active 大概率拖累，但在 16B 可能正向 → 用 A2 时要"外推"而不是直接拿结论
2. **超大 N (256+) 在 1B 上可能 dead expert** → 不代表 16B 也会
3. **20T+ tokens 的退火 / mid-training 现象** A2 看不到（只跑 25B）

所以 A2 给方向，A3/A4 验证 scaling 外推一致。

### 一句话总结

> **Wind tunnel = 用 200M-4B 的缩小版模型按 ladder 跑消融，先验证架构再投入 16B 主训练；A2 (1B/200M) 是核心消融点，T1-T2-T3 三层 9 个消融分批回答关键设计问题。**

参考：`papers/22_FINAL_16B_design.md §8` + `29_wind_tunnel_a2.md`

---

## 10. 工程语汇 — EP / TP / DP / PP / Dropless / Megablocks

### 并行策略 4 件套

| 名字 | 切什么 | 类比 |
|---|---|---|
| **DP** Data Parallel | 切**数据 batch** | 4 张卡分别处理不同 sample，模型完全复制 |
| **TP** Tensor Parallel | 切**单层矩阵** | 把一个大 GEMM 切成 4 块在 4 张卡上算 |
| **PP** Pipeline Parallel | 切**层** | 第 1-15 层在 GPU 0，第 16-30 层在 GPU 1 |
| **EP** Expert Parallel | 切**expert** | 64 个 expert 分到 8 张卡上，每张卡 8 个 |

通常组合：DP × TP × PP × EP 同时用（叫 4D parallelism）。

### Why EP 是 MoE 特有

Dense 模型只能用 DP + TP + PP。MoE 多了一个维度 —— **expert 可以分卡放**。这让 MoE 在 GPU 数量上的 scaling 比 dense 更灵活。

### Dropless 是什么、为什么重要

**老式 MoE (Switch 2021)**：设一个 capacity factor C，每个 expert 最多接 C × (B×S/N) 个 token，超出就 **drop**（丢弃）。
- 优点：每个 expert 的 batch size 固定 → kernel 简单
- 缺点：丢 token = 信息损失，loss 不稳定

**Dropless (Megablocks 2022)**：所有 token 都送（不丢），允许 expert 之间 batch size 不均。
- 优点：无信息损失、quality 显著好
- 缺点：grouped GEMM 形状不规则，kernel 复杂

**现代所有 MoE 都用 dropless**。Megablocks (MIT-IBM 开源) 是事实标准库，提供 `permute` (dispatch) + `gmm` (grouped GEMM) + `combine` 三个核心 kernel。

### All-to-all 是什么

EP 下，token 经过 router 知道要去哪个 expert → 需要把它"发"到那个 expert 所在的 GPU。这个跨 GPU 数据传输叫 **all-to-all collective**：所有 GPU 互相发送 / 接收。

- 单节点 8 H100：NVLink 4.0 带宽 450 GB/s，all-to-all 几乎不是瓶颈
- 跨节点：受 InfiniBand 带宽限制 (100-400 GB/s)，会成为瓶颈 → 所以 V3 用 node-limited routing (M=4) 控制跨节点 token 数

你的 16B 用 EP=8 单节点 → 不需要 node-limited routing（参考 25_node_limited_routing.md）。

### 一句话总结

> **DP/TP/PP 是 dense 通用并行，EP 是 MoE 特有（按 expert 分卡）；Dropless = 不丢 token + 不规则 grouped GEMM，是现代标准；Megablocks 是事实库；EP=8 单节点下 NVLink 带宽不是瓶颈。**

参考：`papers/22_FINAL_16B_design.md §6` + `25_node_limited_routing.md`

---

## 11. MoE 特有的不稳定问题与解药

MoE 训练比 dense 容易出 spike / divergence。最常见 5 个问题：

### 11.1 Dead experts（路由雪球）

- 症状：某些 expert 拿到的 token <0.1% 全部 batch
- 因：第 4 节讲的雪球退化
- 解：aux loss 或 ALF（第 4 节）
- 必加监控：`route/active_experts` = "拿到 ≥1% token 的 expert 数"，期望 ≥ N × 0.9

### 11.2 Gate logits 爆炸

- 症状：sigmoid/softmax 前的 logits 取值 ±50+，gate 完全饱和 → 选择不稳定
- 因：W\_router 没有约束，训久了 norm 一路涨
- 解：**Router z-loss** $L_z = \beta \cdot (\text{logsumexp(logits)})^2$，OLMoE 推荐 β=0.001
- 所有现代 MoE 都加这个

### 11.3 Attention QK 爆炸

- 症状：Q 和 K 的点积 巨大 → softmax 饱和 → 单一 token 接管 attention
- 因：Q, K 的 norm 一路涨
- 解：**QK-Norm** — 在 Q 和 K 上各加一个 RMSNorm
- 几乎所有 2024+ 模型都加（OLMoE, dots1, Qwen3, Hunyuan-A13B, ...）

### 11.4 ALF bias 漂移

- 症状：bias[i] 累积到 ±10 → 路由完全被 bias 主导，gate 失去意义
- 因：sign 更新长期不平衡（如果某些 expert 长期 overloaded）
- 解：Ling 2.0 零均值修正：`b ← b − γ·(sign(error) − mean(sign(error)))` → 强制 bias 均值不漂
- 不用零均值修正的 dots1 在 11.2T tokens 上也跑通了 → 这个修正非必须

### 11.5 Gradient spike at end of warmup

- 症状：warmup 结束时 LR 跳到 peak → grad norm 跳到 10×
- 因：MoE 路由对 LR 敏感
- 解：用 **WSM (Warmup-Stable-Merge)** scheduler 而不是 cosine，末段权重平均吸收震荡（Ling 2.0 §2.4）
- 或 fallback：WSD (Warmup-Stable-Decay)，dots1 / V3 同款

### 一套必加的稳定性"四件套"

- ✓ **Router z-loss** β=0.001
- ✓ **QK-Norm**
- ✓ **FP32 gating**（router 计算用 FP32，不走 BF16）
- ✓ **Embedding weight decay**（OLMoE 强调）

这四样几乎不需要消融，**直接默认开**。

### 一句话总结

> **MoE 5 种典型不稳定 = dead experts / gate 爆 / QK 爆 / bias 漂 / spike-on-warmup-end；解药四件套 = aux loss 或 ALF + Router z-loss + QK-Norm + FP32 gating，直接默认开。**

参考：`papers/09_olmoe.md §4.1.7` + `24_dots1.md §2`

---

## 12. 把这些回到 22_FINAL spec — 每个决策的因果链

讲完所有概念，回头看 22_FINAL Profile B 的每个具体数字，你应该能用第 1-11 节的因果链解释为什么这么选。

### 总参 15.5B
**因果**：单卡 H100 / L40S 部署得下（FP16 下 ~32GB 模型 + KV cache + activation）；< 14B 偏 reasoning（密度更高）；> 20B 推理 latency 翻倍 + 双卡 serving。**目标"大但便宜"的 sweet spot**。

### Active 2.4B
**因果**：推理 FLOPs/token = 2.4B → latency 适中（参考 V2-Lite / Moonlight 已验证）。**严格口径**；V3 口径报 2.9B。

### 27 层 / hidden 2048
**因果**：V2-Lite 同款（DeepSeek 在 5.7T tokens 上验证）。层数 / hidden 比例符合 dense 模型常用 scaling，且 hidden=2048 → head\_dim=128 自然 16 Q-heads。

### N=64 / K=8 / 1 shared expert
**因果**：
- N=64 = 现代 fine-grained 甜区（第 5 节）；> 256 工程复杂度大
- K=8 = 2025 主流（V3 / Ling / K2 / GLM-4.5 共识，比 K=2/6 偏 reasoning）
- 1 shared = 国产派路线（第 6 节），还在 wind tunnel 验证

### Sigmoid + ALF
**因果**：第 4 节解药 2；2024-08 后新模型增量主体；K2 / V3 / Ling / GLM-4.5 共识

### GQA 16Q/4KV（不用 MLA）
**因果**：第 7 节 — MLA 需要 DeepSeek 全套 kernel 生态，16B KV cache 不是瓶颈，ROI 不足；GQA 是 16B 默认

### MTP D=1
**因果**：第 8 节 — V3 风格 causal chain，2.4B active 处于 boundary case，**仍在 wind tunnel A2 T3.1 验证**

### BF16 训练（不用 FP8）
**因果**：FP8 在 V3 671B 上 ROI 高（kernel 大、bandwidth 紧）；16B 上 BF16 已经能撑住，FP8 ops 复杂度不值

### AdamW (β=0.9/0.95, ε=1e-8)
**因果**：行业默认；但 ε 是 V3 派 1e-20 还是 OLMoE 派 1e-8 仍未答 → wind tunnel A2 T1.1 验证

### 14-16T tokens 训练
**因果**：Yokota 2025 警告 over-train (>20T) 会伤 reasoning；K2 在 15.5T 表现好；V3 14.8T；Ling 20T+ 走 reasoning RL 路线。**16T 是不冒险的上限**。

### 第 0 层 dense FFN
**因果**：V2-Lite / DeepSeekMoE / GLM-4.5-Air / ERNIE 4.5 21B 共识。让 first layer 处理低层特征（subword-level），不需要 expert specialization；提高稳定性。

### 一句话总结

> **22_FINAL Profile B 的每个数字背后都对应着 §1-11 的一条因果链；很多是"行业共识 + 2-5 个 SOTA 验证"，少数仍是"开放问题 → wind tunnel A2 来答"。**

---

## 13. 如果只记 5 件事

如果你只想记 5 件事来给同事讲清楚 MoE 是什么：

1. **MoE = 让"总参数"和"推理算力"解耦** → Active 决定推理 FLOPs，Total 决定知识容量；这是 MoE 唯一不可替代的价值
2. **Router 是 MoE 的灵魂也是软肋** → 没有标签的学习；天然倾向于雪球退化 (dead experts)，所以必须配 load balance 机制
3. **Load balance 两条路线** → aux loss (Switch/Mixtral) 把均衡和质量耦合；ALF bias (V3+) 解耦它们；2024-08 后增量主体是 ALF
4. **Fine-grained 革命** → DeepSeekMoE 2024-01 证明"很多小 expert 比少数大 expert 好"，把 MoE 从 N=8 推到 N=64-256；这是 V3 / Ling / Qwen3-30B 一切现代设计的起点
5. **Wind tunnel 不是可选项** → 单次 16B 训练 1.5M 美元，错一个超参就 sunk；A0-A4 ladder + A2 上 9 个消融 + ~5K H100-hr 是"风险对冲"，不是浪费

---

## 14. 还想看哪个细节

读完此文后，按兴趣去对应深度笔记：

| 想深挖什么 | 去看哪个 |
|---|---|
| ALF 论文细节 | `papers/03_auxloss_free.md` |
| Fine-grained 数学 | `papers/01_deepseekmoe.md` + `17_finegrained_scaling.md` |
| V3 完整决策链 | `papers/04_deepseek_v3.md` |
| MTP 因果 + 决策 | `papers/20_mtp_gloeckle.md` + `23_mtp_investigation.md` |
| Attention 变种（MLA / GQA） | `papers/02_deepseek_v2.md` (MLA) + `05_qwen3.md` (GQA) |
| Hybrid attention（Mamba） | `papers/15_jamba.md` + `13_minimax_01.md` |
| 你的 16B 完整 spec | `papers/22_FINAL_16B_design.md` |
| 全市场 60+ MoE 对比表 | `papers/28_open_source_moe_catalog.md` |
| Wind tunnel A2 实验矩阵 | `papers/29_wind_tunnel_a2.md` |
| Router 实施代码细节 | `papers/30_routing_implementation.md` |
