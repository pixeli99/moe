# Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints

- **arXiv**: 2212.05055 (v2: 17 Feb 2023, 原 v1: 9 Dec 2022)
- **发表**: ICLR 2023
- **机构**: Google Research（除 Aran Komatsuzaki 当时为 Georgia Tech 实习生）
- **作者**: Aran Komatsuzaki\*†, Joan Puigcerver‡, James Lee-Thorp‡, Carlos Riquelme, Basil Mustafa, Joshua Ainslie, Yi Tay, Mostafa Dehghani, Neil Houlsby
- **代码**: github.com/google-research/vmoe (vision), github.com/google-research/t5x/tree/main/t5x/contrib/moe (语言)

## TL;DR
**Sparse Upcycling**：用已有 dense checkpoint **温启动** MoE 训练，省去从头训 MoE 的初期成本。

核心配方：
1. 把 dense FFN 层 **复制 E 份**作为 E 个 expert（每个 expert 与原 FFN 完全相同）。
2. **Router 随机初始化**（这是从无到有的唯一部分）。
3. 其余所有参数（embedding、attention、layernorm、unembedding）从 dense checkpoint 直接复制。
4. 用 dense 的训练超参数（同 batch / lr schedule / weight decay）继续训。

效果：在 T5 (Base/Large/XL) 和 ViT (B/16, L/16) 上验证，**用 ~50% 的初始 dense 训练成本**就能让 upcycled MoE 显著超过其 dense 起点。

## 核心命题
1. **Upcycling 在 [+10%, +60%] 的额外 budget 区间最划算**——比继续训 dense 强很多，比 from-scratch MoE 强一倍多。
2. **From-scratch MoE 需要约 120% 的初始 dense 训练 budget 才能追平 upcycled MoE**——即"upcycling 等于免费送 120% 的训练时间"。
3. **极长训练下，from-scratch MoE 会最终反超 upcycled**——upcycling 是 budget-constrained 才优。
4. **Router 必须从头训**（这是 surgery 的"伤口"），但 expert 用复制粘贴的 FFN 比随机初始化好得多。

## 关键公式 / 关键设置

### Upcycling 算法（§3）
- 假设原 dense Transformer 有 dense FFN 层数 = L_dense。
- 选其中 L_MoE / L_dense（通常一半）的层替换为 MoE。
- 每个 MoE 层：
  - **E 个 expert** ← 每个 expert = 原 FFN 权重的精确拷贝（identical）。
  - **Router** ← 随机初始化的小线性层。
  - **其他**: layer-norm、attention 不变。
- 训练时所有参数 unfrozen，用原 dense 的优化器配置继续。

### 关键超参数（recipe）
- **Router type**: Expert Choice (C=2) for vision + T5 encoder；T5 decoder 用 Top-K (K=2) 保证 inference 一致性。
- **Number of MoE layers**: 替换最后 50% 的 FFN（如 12 层模型替换 6 层）。
- **Number of experts (E)**: **E = 32 是良好的折衷**（vision），E=128 也试过。
- **Capacity factor (C)**: C = 2 在 train-time vs 质量上最优。
- **Resume optimizer state**（Adam moments）: **vision 复用有用，语言上没区别**。
- **Normalize router weights after routing**: vision 上有用，语言上有害。
- **Expert noise injection**: 小幅 OK，大幅有害。

### FLOPs 模型
- Expert Choice routing 下，capacity = T = C(n/E)，FLOPs 与 dense 几乎相同（router 开销可忽略）。
- 多 expert 几乎不增加 FLOPs。

## 实验设置

### Vision 实验（V-MoE）
- 基座：JFT300M pretrain 的 ViT-B/32, B/16, L/16。
- ImageNet 10-shot 评估（5 个不同 training sets 求均值）+ ILSVRC2012 full finetune。
- E = 32 experts，C = 1（per step 比较）；6 layers MoE out of 12。
- Dense checkpoint 训 14 epochs，upcycle 后再训 7 epochs（共 21 epochs）。
- Global average pooling，Expert Choice 路由。

### 语言实验（T5）
- T5 Base / Large / XL（用 T5 1.1 official checkpoints）。
- Span-corruption pretrain on C4，downstream finetune on SuperGLUE。
- E = 32, C = 2, 6 MoE layers interspersed。
- 训 0.5M – 1M extra steps。

## 主要结论

### 1. 上限 budget 区间：upcycling 完爆 continued dense（Fig. 2）
- Vision (JFT validation precision)：ViT-L/16 upcycled 在 1e2 TPU-core-days 时已达 58%，dense continuation 远不及。
- 语言 (C4 token accuracy)：T5-XL upcycled 在 1e2 TPU-core-days 时达 ~75%（曲线明显高于 dense）。
- **Abstract 原话**: "upcycled T5 Base, Large, and XL language models and ViT B & L models significantly outperform their dense counterparts on SuperGLUE and ImageNet, using only ~ 50% of the initial dense pretraining sunk cost"。
- **Section 2 关键数字**: 
  - ViT-B/16 提升 1% on ImageNet 10-shot：dense continuation 需要 **+58% extra training**；upcycling **只要 +13%**。
  - T5-Large/Base 在 SuperGLUE 上比 dense counterpart 高 **1.5–2 个绝对点**，分别用 46% / 55% extra training。

### 2. 与 from-scratch MoE 的对比（Fig. 4）
- 语言任务：**from-scratch MoE 要训约 120% 原 dense budget 才能追上 upcycled**（Fig. 4 中第二靠右的橙色和绿色点）。
- 但是，**继续训下去 from-scratch 最终会反超**（Fig. 4 趋势线斜率更陡）。
- 论文原文："Figure 4 suggests that, given a very large computation budget (> 100% of the initial dense model's computation budget), the MoE-from-scratch model will eventually catch the upcycled model"。
- **拐点经验法则**：if extra budget < 100% of original dense → upcycle；else → scratch。

### 3. Warm-start（depth tiling）作为额外 baseline（Fig. 5）
- Warm-start：复制层 + 加深（Rae et al. 2021 "depth tiling"）。
- Warm-start 比 dense continuation 好，但**显著不如 upcycling**。

### 4. 初始 dense 训练成熟度的影响（Fig. 6）
- 在不同 dense pretrain checkpoint（200k / 400k / ... / 1.2M steps）上 upcycle 再训 200k steps。
- **Upcycling 的增益与 dense 起点训练成熟度无关**——dense 不论训多久，upcycle 都加 ~固定增量。

### 5. Ablations
- **Number of experts**: E=32 综合最优。E>>32（vision）会让 initial quality drop 更深，需要更多 budget 才能恢复。
- **Expert init**: 复制 dense FFN > random init（除非 budget 极大）。
- **Adding noise to copied experts**: 小幅 OK，大幅有害——太多噪声破坏 dense 知识。
- **Router init**: 随机（zero init 会让 router 退化）。

## 对 16B MoE 设计的启示

### 是否对 16B 用 upcycling？

**典型 16B MoE 训练 budget 远超过 dense 等参数 checkpoint**（如 LLaMA-2 7B 用 ~2T tokens；要求的 16B MoE 训 ~1T+ tokens）。
- 如果存在 high-quality 2B–7B dense checkpoint（如 LLaMA-3 8B、Qwen2.5-7B）：
  - **Upcycle 节省初期成本是显著的**。Mixtral 8x7B 实际上就是用 Mistral 7B upcycle 的。
- 但 16B MoE 通常需要训 > 200B tokens，已经接近或超过 "from-scratch 反超" 的临界点。
- **推荐策略**：
  - **若总 budget < 1× dense 训练成本 → upcycle**（适合"快速出 MoE 版本"场景）。
  - **若总 budget ≥ 1.5× dense → from scratch + careful router init**。
  - **混合**：Upcycle 起手 + 长期训练 + 中段加 expert noise 让 expert 分化（DeepSeek 等用过类似策略）。

### Upcycling 的关键配置（如果选 upcycle）
- **E = 8 或 16**（比 Komatsuzaki 用的 32 少；与现代 MoE 实践对齐）。
- **Router 随机初始化** + 1k-2k step warmup。
- **复用 optimizer state**：在语言任务上效果中性，但**不会有害**。在 large-batch + Adafactor 下尤其推荐。
- **替换 50% 的 FFN 为 MoE**（隔层 MoE 是最常见做法）。如果用 fine-grained MoE，可能需要替换更多。
- **Capacity factor C = 1.0 – 1.25**（论文用 C=2 是上一代默认值；现代 dropless 实现可以 C=1）。
- **Expert init 加小噪声**（Gaussian σ ≈ 0.01）促进分化，noise 太大有害。

### Granularity (G) 与 upcycling 的冲突
- Krajewski 推荐 G > 1（fine-grained）。但 upcycling 时 expert = 复制的 FFN（G=1 by construction）。
- **若要 G > 1 + upcycling**：把原 FFN 沿 hidden dim 切分成 G 个小 chunk，每个 chunk × E 份。Komatsuzaki 论文未做过这件事——这是开放方向。
- 简单实务：**对 16B，第一阶段先 upcycle (G=1, E=8)；中段再做一次 surgery 切分到 G=4 + 短训稳定**。

### 与 1/8 sparsity 的关系
- Komatsuzaki 测的 E=32, top-K = 2 → S = 30/32 = 0.94。比 1/8 sparsity (S=0.875) 还稀疏。
- 对 16B：**E=8 K=1（S=0.875）兼容上述 upcycle recipe**——把 1 个 FFN 复制 8 份，router init from scratch，top-1 routing，即为 Mixtral 风格。

### 一句话推荐
**若 16B 训练 budget 在 200B–500B tokens 之间且有 7B-8B dense checkpoint 可用，Sparse Upcycling 是首选起手式；Router 随机 + 复制 FFN + 加 σ=0.01 噪声 + Adam 状态可选**。

## Caveats / 局限

1. **长训练下 from-scratch MoE 反超**：上 1T+ tokens 训练，scratch 会胜出。
2. **Expert 在 upcycle 初期高度相关**——分化需要时间和合理的噪声 / 数据多样性。
3. **Router 性能取决于初期 load balancing**：随机 router 可能在初期发生"专家坍缩"，需要 z-loss / load-balance loss 配合。
4. **本文用的是 T5 (encoder-decoder) 和 ViT (vision)，不是 decoder-only LLM**——但 Mixtral / DeepSeek 等实践证明 recipe 大致迁移得过去。
5. **没探索 fine-grained (G>1) + upcycling 的组合**——这是 16B MoE 设计中的 open question。
6. **Capacity factor C=2 是旧时代设置**：现代 dropless 实现下 C=1 就够。
7. **没有 instruct / RL phase 实验**。
8. **大 E（E>>32）时初期质量 drop 大**：可能需要 router warmup + frozen experts 短期训练破解。
