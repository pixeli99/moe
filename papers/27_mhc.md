# mHC: Manifold-Constrained Hyper-Connections

- **arXiv**: 2512.24880 (v2, 2026-01-05；v1 2025-12-31)
- **机构**: DeepSeek-AI
- **代码**: 未公开（TileLang 内部 kernel）
- **作者**: Zhenda Xie*†, Yixuan Wei*, Huanqi Cao*, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang

## TL;DR

**Hyper-Connections (HC, Zhu 2024 ByteDance/CRO)** 把残差流从 $C$ 维扩到 $n \times C$ 维（典型 $n=4$），引入 3 个学习映射 $\mathcal{H}^{\text{pre}}, \mathcal{H}^{\text{post}}, \mathcal{H}^{\text{res}}$ 实现"多流残差"。**问题**：composite $\prod \mathcal{H}^{\text{res}}$ 不受约束 → 27B 模型上 forward Amax Gain 飙到 **3000×**、12k step loss 突然 surge → HC 训练不稳。

**mHC 的解**：把 $\mathcal{H}^{\text{res}}$ **投影到 doubly stochastic 矩阵流形（Birkhoff 多面体）**，用 Sinkhorn-Knopp 算法（20 次迭代）。doubly stochastic 矩阵谱范数 ≤ 1，且对矩阵乘法封闭 → composite 永远稳定。$\mathcal{H}^{\text{pre}}$ 加 sigmoid、$\mathcal{H}^{\text{post}}$ 加 $2\sigma$ 防 cancellation。配合 TileLang custom kernel + selective recompute + DualPipe overlap，**$n=4$ 训练 overhead 控制在 6.7%**。

**关键定位**：mHC **不是独立 add-on**，是 **HC 的修复补丁**。如果你不打算用 HC 多流残差，mHC 就解决不了你的问题；如果用了 HC，mHC 让它在大规模训练上不发散。

## 关键数学

### HC 复习（Eq. 3，Zhu 2024）

把残差从 $C$ 维扩到 $n \times C$ 维：

$$\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)$$

其中：
- $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ —— mix 多流残差
- $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$ —— 把 $n \times C$ 流聚合成 $C$ 维 layer 输入
- $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$ —— 把 layer 输出 broadcast 回 $n$ 流

### HC 不稳的证据（Fig. 2-3）

27B model 上：
- $\prod \mathcal{H}^{\text{res}}$ 的 **Amax Gain Magnitude** 在 layer 30 处 peaks at **3000**（forward signal 爆炸）
- HC 在 12k step 出现 unexpected loss surge
- gradient norm 在同一时刻 spike

**问题根源**：$\mathcal{H}_l^{\text{res}}$ 是 unconstrained learnable matrix，composite $\prod \mathcal{H}_{L-i}^{\text{res}}$ 失去全局 mean conservation，signal 在深度方向无界放大。

### mHC 修复（Eq. 6-9）

**核心约束**：$\mathcal{H}^{\text{res}}$ 投影到 doubly stochastic 矩阵流形 $\mathcal{M}^{\text{res}}$（Birkhoff 多面体）：

$$\mathcal{P}_{\mathcal{M}^{\text{res}}}(\mathcal{H}_l^{\text{res}}) := \{\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n} \mid \mathcal{H}_l^{\text{res}} \mathbf{1}_n = \mathbf{1}_n,\ \mathbf{1}_n^\top \mathcal{H}_l^{\text{res}} = \mathbf{1}_n^\top,\ \mathcal{H}_l^{\text{res}} \geq 0\}$$

即：行和 = 列和 = 1，元素 ≥ 0。

**3 大数学性质（论文 §4.1）**：
1. **Norm preservation**：谱范数 $\|\mathcal{H}_l^{\text{res}}\|_2 \leq 1$ → 防梯度爆炸
2. **Compositional closure**：doubly stochastic 矩阵在乘法下封闭 → $\prod \mathcal{H}^{\text{res}}$ 也是 doubly stochastic → 全深度稳定
3. **Birkhoff polytope 几何**：是 permutation matrices 的 convex hull → residual mapping = permutations 的 convex combination → 单调增加跨流 mixing

### 完整公式（Eq. 7-9）

```
x'_l = vec(x_l)                                       # flatten n×C → 1×nC
H̃^pre = α^pre · (x'_l φ^pre) + b^pre                  # dynamic + static
H̃^post = α^post · (x'_l φ^post) + b^post
H̃^res = α^res · mat(x'_l φ^res) + b^res               # 注意 mat() reshape

H^pre = σ(H̃^pre)              # sigmoid → 非负
H^post = 2σ(H̃^post)            # 2·sigmoid → 非负，乘 2 抵消缩放
H^res = Sinkhorn-Knopp(H̃^res)  # 投影到 doubly stochastic
```

**Sinkhorn-Knopp 算法**（Eq. 9）：
- $M^{(0)} = \exp(\tilde{\mathcal{H}}^{\text{res}})$ —— exponentiation 保证非负
- 迭代 $M^{(t)} = \mathcal{T}_r(\mathcal{T}_c(M^{(t-1)}))$ —— 交替行列归一化
- $t_{\max} = 20$（论文实测够用）

## 系统设计（论文 §4.3）—— 6.7% overhead 的来源

### Kernel Fusion（§4.3.1，Eq. 10-19）

**自定义 3 个 mHC kernel** 在 TileLang 中实现：

| 阶段 | 操作 | 精度 | I/O 优化 |
|---|---|---|---|
| K1 | $\tilde{\mathcal{H}}^{\text{pre/post/res}} = \tilde{x}_l \varphi_l$ + RMSNorm 内嵌 | FP32 (compute), BF16 (input) | 两次 scan on $\tilde{x}_l$ fused into 1 GEMM kernel |
| K2 | $\sigma(), 2\sigma(), \text{Sinkhorn}$ | FP32 | 3 个轻量操作 fused |
| K3 | apply mappings + residual merge | mixed | reads from $(3n+1)C$ → $(n+1)C$ |

**RMSNorm 重排**：把 divide-by-norm 操作 **移到 GEMM 之后**而非之前，等价但允许更大 tile size（reduce launch overhead）。

### Recomputing（§4.3.2）

mHC 中间激活在 forward 后 discard，backward 时**重算 mHC kernels** —— 所有 kernel 已 fused，重算成本可控。这关键到能用 gradient checkpointing 的相同 budget 撑住 $n=4$ 的 4× 残差宽度。

### DualPipe 通信 overlap（§4.3.3）

HC 在 PP 中需要传 $n \times C$ 维 hidden state（4× standard），naive PP 通信成本高。mHC 把 communication carefully 拆分到 DualPipe 的 forward / backward chunks 里，与 GEMM 计算 overlap，**6.7% wall-clock overhead 是这一系列优化后的最终值**。

## I/O 对比（来自 AttnRes 论文 Table 1）

每 token 每层 residual 机制 I/O cost：

| 方案 | Read | Write | **总 I/O (typical)** |
|---|---|---|---|
| 标准 residual | 2d | d | **3d** |
| **mHC (m=4)** | $(8m+2)d + 2m^2 + 4m$ | $md$ | **34d** |
| AttnRes Full | $(S+N)d$ | $2d$ | 24d (S=16, N=4) |
| AttnRes Block | $(N/S)d + 3d$ | $2d$ | **5.5d** (N=8) |

**mHC 比标准残差贵 11×**。在 16B 这种 HBM bandwidth 已经吃紧的尺寸上，这是**真实 wall-clock 成本**，不是论文的 6.7% overhead 那么轻松（6.7% 是与 HC 比，不是与 vanilla residual 比）。

## 实验模型配置（Table 5，3 档 + 锚点；2026-01-05 v2 paper）

DeepSeek mHC 论文真正的实验锚点 ladder（**与 22_FINAL_16B_design wind tunnel 直接可对比**）：

| 档位 | Total | Active | Layers | Hidden | FFN (expert) | Routed | TopK | Shared | Attention | Q heads | KV rank | head_dim | Vocab | Seq | Batch | Steps | Tokens | Peak LR | Warmup | n (HC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **3B 标准** | **2.97B** | **612M** | **12** | **1280** | **896** | **64** | **6** | **2** | **MLA** | 16 | 512 | 128 | 129280 | 4096 | 320 | 30k | **39.3B** | 8.6e-4 | 2000 | 4 |
| **3B 长训** | 2.97B | 612M | 12 | 1280 | 896 | 64 | 6 | 2 | MLA | 16 | 512 | 128 | 129280 | 4096 | ? | 100k | **1.05T** | 9.0e-4 | ? | 4 |
| 9B | 9.18B | 1.66B | 18 | 1920 | 1280 | 64 | 6 | 2 | MLA | 24 | 512 | 128 | 129280 | ? | 512 | 50k | 105B | 5.9e-4 | ? | 4 |
| **27B（主对比）** | **27.0B** | **4.14B** | **30** | **2560** | **1536** | **72** | **6** | **2** | **MLA** | 32 | 512 | 128 | 129280 | ? | 1280 | 50k | **262B** | 4.0e-4 | ? | 4 |

**Optimizer**：AdamW (β1=0.9, β2=0.95, ε=**1e-20**, wd=0.1) —— **ε 极小是 DeepSeek 全家配方特征**。
**Infra**：DualPipe PP schedule；论文未明说 FP8。
**Sinkhorn-Knopp iter**：20（所有档位一致）。

### 与本仓库 16B 主 spec 的可比性

| 维度 | mHC 3B (V3-mini) | mHC 27B | 22_FINAL Profile B (16B) |
|---|---|---|---|
| Routed / Active / Shared | 64 / 6 / 2 | 72 / 6 / 2 | 64 / 8 / 1 |
| 激活率（active/total） | 20.6% | 15.3% | 15.5%（严格口径 2.4B/15.5B） |
| Hidden / Layers | 1280 / 12 | 2560 / 30 | 2048 / 27 |
| Attention | MLA | MLA | GQA 16Q/4KV |
| FFN expert dim | 896 | 1536 | 1408 |
| Vocab | DS 129280 | DS 129280 | 128K BBPE |
| Sparsity 路线 | V3-style | V3-style | V2-Lite 现代化 + Yokota K=8 |

→ **mHC 3B ≈ "缩小版 V3 + HC"**；本仓库 wind tunnel A2 如果对标 mHC 3B（39B tokens，2.4× active 比例），可以直接复用其超参（LR=8.6e-4 / β2=0.95 / ε=1e-20 / batch=320）作为初始 sweep 中心。**注意 ε=1e-20 与 OLMoE 推荐的 1e-8 差 12 个数量级 —— 这是 DeepSeek 的特色，OLMoE 路线请用 1e-8。**

### 与 22_FINAL §8 wind tunnel A2 (1B/200M-active/25B) 的关系

- mHC 3B 标准档（612M active, 39.3B tokens）是 **A2 配置的 ~3× active、~1.5× tokens** 的上限版
- 建议：A2 baseline 抄 mHC 3B 的优化器超参，但 active 缩到 200M、tokens 缩到 25B、attention 改 GQA（标准残差），作 D=0 vs D=1 (MTP) 对照
- mHC kernel 因 TileLang 不开源、I/O 11× vanilla residual，**A2 不引入 mHC/HC 变量**

## 稳定性证据（Fig. 2，27B model）

| 指标 | HC | mHC |
|---|---|---|
| Loss surge @ 12k step | **有**（明显的 spike） | **无** |
| Gradient norm @ 50k step | 0.15-0.25 区间剧烈震荡 | 平滑下降到 0.05 |
| Composite $\prod \mathcal{H}^{\text{res}}$ Amax Gain | peaks 3000 | 接近 1（bounded） |

### Ablation（Table 1）

逐项启用 HC 三映射的 absolute loss gap（baseline = vanilla residual）：

| $\mathcal{H}^{\text{res}}$ | $\mathcal{H}^{\text{pre}}$ | $\mathcal{H}^{\text{post}}$ | Loss Gap |
|---|---|---|---|
| – | – | – | 0.0 |
| ✓ | – | – | **−0.022** |
| ✓ | ✓ | – | −0.025 |
| ✓ | ✓ | ✓ | −0.027 |

→ $\mathcal{H}^{\text{res}}$ 是大头，pre/post 加一起还多 0.005。说明 mHC 真正核心是 res mapping 的 manifold 约束。

### vs AttnRes（来自 AttnRes 论文 Table 2 的独立验证）

5 个 anchor 模型（194M-528M active），mHC(-lite) vs Block AttnRes vs Full AttnRes：

| Active | Baseline | Block AttnRes | Full AttnRes | **mHC(-lite)** |
|---|---|---|---|---|
| 194M | 1.931 | 1.909 | **1.899** | 1.906 |
| 241M | 1.895 | 1.875 | 1.874 | **1.869** |
| 296M | 1.829 | 1.809 | **1.804** | 1.807 |
| 436M | 1.766 | 1.746 | **1.737** | 1.747 |
| 528M | 1.719 | 1.693 | **1.692** | 1.694 |

→ **三方几乎打平，mHC 不在 loss 上占优**；I/O 上 mHC **比 Block AttnRes 贵 6×**。

## 与本仓库的契合度（16B MoE 适配）

### 不推荐用于 16B 的理由

1. **依赖 HC**：mHC 是 HC 的稳定性补丁，**先采用 HC 才有 mHC 的"价值"**。本仓库 22_FINAL_16B_design 没有 HC，加 mHC = 同时引入两个新结构 + 4× 残差宽度
2. **I/O 11× 更贵**：mHC m=4 = 34d/token 比 vanilla 3d 高 11×。16B 是 HBM-bound 的尺寸，不像 671B+ 那样 attention/FFN compute 主导
3. **6.7% overhead 是 vs HC 的**：vs vanilla residual 实际开销远高（~30%+），论文没明给
4. **kernels 不开源**：TileLang custom kernel + DualPipe overlap 都是 DeepSeek 内部 stack，复现成本高
5. **没有下游 benchmark 数据**：论文只给 loss 和稳定性 plots，没给 MMLU/GSM8K 等 evaluation 表
6. **Kimi head-to-head 没赢**：5 个 anchor 上 mHC(-lite) 与 AttnRes 打平甚至略输

### 唯一会想用 mHC 的场景

- **如果团队已经在用 HC** 且观察到了 12k step 风格的 loss surge → mHC 是当下最直接的稳定化方案
- **如果模型规模 > 200B 且 attention/FFN compute 占绝对主导** → mHC 的 I/O 增量被稀释，相对成本下降
- **如果团队 infra 是 DeepSeek 体系**（TileLang + DualPipe）→ 复用 mHC kernel 成本低

**16B + 标准 PreNorm + GQA + 64 expert 的 spec 都不在这三个场景里**。

## Caveats / 局限

- **2025-12 才发布**，目前没有任何独立第三方复现
- **27B 是论文唯一的训练规模**（Fig. 2-3 的 stability plots），没有 100B+ 验证
- **完整 ablation 缺失**：没有 mHC vs 标准 residual 的下游 benchmark 对比，只有 vs HC 的 loss surge 对比
- **6.7% overhead 是 n=4 with full kernel optimization** 的最优情况；如果团队没有 TileLang 工具链，实际开销远更高
- **DualPipe 依赖**：communication overlap 需要 DualPipe 风格 PP 调度。1F1B-only 的 framework 适配难度大
- **Sinkhorn-Knopp 20 iter 是 fixed**，没消融到底多少 iter 是 sufficient/optimal
- **Birkhoff 多面体的几何论证**虽然漂亮，但**没回答"为什么不直接用 orthogonal matrix manifold (Stiefel)"** —— orthogonal 也保 norm 但没要求 doubly stochastic
- **HC 是 ByteDance 2024 的工作，独立应用不广泛** —— mHC 在 HC 已经不是主流的当下推出，受众范围本身有限

## 与本仓库的交叉引用

- **26_attention_residuals.md**：直接竞争方案，I/O 5.5d vs mHC 34d；scaling law 上打平；推荐 16B 选 AttnRes 而非 mHC
- **22_FINAL_16B_design.md** §11："强默认但 pilot 必测"列表里 **mHC 不进入推荐**，仅作为 §13 caveats 的"已调研但不采用"
- **04_deepseek_v3.md**：V3 没用 HC 也没用 mHC；DeepSeek 自己的旗舰模型也没把 HC/mHC 列为标准配置
- **08_ling_2.md** + **24_dots1.md**：Ling 和 dots1 都是标准 PreNorm + 标准 residual，没有走 HC 路线
