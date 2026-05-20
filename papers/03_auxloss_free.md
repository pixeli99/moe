# Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

- **arXiv**: 2408.15664
- **机构**: DeepSeek-AI / Peking University
- **发表时间**: 2024 年 8 月
- **作者(代表)**: Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai

## TL;DR
本论文指出 MoE 训练中长期使用的 **auxiliary balance loss**（即 α·Σ f_i·P_i）会给所有 token 引入额外的"非语言任务梯度"，造成性能 vs 负载均衡的根本性 trade-off。作者提出 **Loss-Free Balancing**：给每个专家引入一个 **不参与 gating 权重计算**的偏置 b_i，b_i 仅参与 top-K 选择；每个 batch 后用 **b_i ← b_i + u · sign(c̄_i − c_i)** 简单地把"过载/欠载"反馈进偏置。这样 router 的训练梯度完全不被 balance 项干扰，同时获得显著更好的均衡度。在 **1B/3B（DeepSeekMoE 架构、64 routed + 2 shared、K=6）** 规模、100B/200B tokens 实验上，Loss-Free 比 Loss-Controlled 在验证集 perplexity 上略好（9.50 vs 9.56 @1B；7.92 vs 7.97 @3B），同时 MaxVio_global 降低 **13–18×**。这套方法被直接搬进 DeepSeek-V3，是后者抛弃 expert-/device-/communication-balance loss 的理论依据。

## 关键方法

### 1. 偏置项 b_i 的位置（Eq. 3）
路由公式被改成：
$$
g_{i,t}=
\begin{cases}
s_{i,t}, & \text{if } s_{i,t}+b_i \in \text{TopK}\!\big(\{s_{j,t}+b_j\}_{j=1}^{N},\;K\big)\\
0, & \text{otherwise}
\end{cases}
$$

**关键点**：
- b_i **只用于 top-K 选择**，最终 gating 权重仍是原始的 s_{i,t}（softmax 后的亲和度），**不**乘 b。
- 这样：(a) 不改变被选中专家的有效权重；(b) 不污染 router 的梯度方向。
- b_i **不参与反向传播**（不是参数，是一个外部状态）；它由下面的规则更新。

### 2. b_i 的更新规则（Algorithm 1）
对每个 batch：
1. 用当前的 g_{i,t} 训练模型；
2. 统计 batch 内每个专家被路由到的 token 数 c_i 与理论均衡量 c̄_i；
3. 计算违规误差 e_i = c̄_i − c_i（正：欠载；负：过载）；
4. 更新偏置：
   $$
   b_i \leftarrow b_i + u \cdot \text{sign}(e_i)
   $$
   其中 **u = 0.001**（也即 V3 称的 γ）。

变体 b_i ← b_i + u·e_i（不用 sign）会让 MaxVio 略低，但 perplexity 同样或略差（Table 3）；最终采用 sign 版本。

### 3. MaxVio 度量（Eq. 4）
$$
\text{MaxVio} = \frac{\max_i \text{Load}_i - \overline{\text{Load}}}{\overline{\text{Load}}}
$$
其中 Load_i 是专家 i 实际接收 token 数，分母是均匀理想值。论文同时报告：
- **MaxVio_global**：整 validation set 累积统计
- **MaxVio_batch**：单 batch 内瞬时最大违规

### 4. 与传统 aux-loss 的本质区别
传统：L_balance = α · Σ f_i · P_i（Eq. 2）。该项对每个被路由的 token 都贡献一个非语言模型梯度，把 router 的训练目标从"选最有用的专家"扭向"让 f_i 更平均"。α 必须很小才不破坏建模性能，但太小又控制不住负载。
Loss-Free：用一个**不可学的、批后更新的偏置**充当反馈控制器（类似 PI controller 中的 integrator），与 router 的训练梯度完全解耦。

## 关键消融与实验结果

### Table 2 — 主结果（验证集 perplexity / MaxVio_global）
| Model size | Method | Perplexity ↓ | MaxVio_global ↓ |
|---|---|---|---|
| **1B** | Loss-Controlled (α=tuned aux-loss) | 9.56 | 0.72 |
| **1B** | **Loss-Free (u=0.001)** | **9.50** | **0.04** |
| **3B** | Loss-Controlled | 7.97 | 0.52 |
| **3B** | **Loss-Free (u=0.001)** | **7.92** | **0.04** |

→ **更好的 perplexity** + **18× 更稳的均衡度**。

### Figure 2 — α 扫描的 trade-off 曲线
在 Loss-Controlled 中 α 太小则失衡严重，α 太大则 perplexity 上升；不存在两全的 α。Loss-Free 直接跳出这条曲线。

### Table 3 — sign vs 直接 e_i
| Update rule | Perplexity | MaxVio_global |
|---|---|---|
| sign(e_i), u=0.001 | 9.50 | 0.044 |
| e_i, u=0.01 | 9.53 | 0.028 |

sign 版本性能略好，且更稳；e_i 版本均衡度略好但 perplexity 略差。论文选 sign。

### Table 4 / Figure 4 — u 扫描
- u=1e-4：早期收敛过慢；
- **u=1e-3：最佳**；
- u=1e-2：后期偏置振荡，性能下降。

### 关于"乘性偏置"的消融
试过把偏置改为乘性（s_{i,t} · b_i）：性能略差，均衡无明显改善 → 采用加性。

## 模型 / 训练规模

- **1B 模型**：DeepSeekMoE 架构，9 个 MoE 层，64 routed + 2 shared 专家，K=6 activated；100B tokens。
- **3B 模型**：11 个 MoE 层，同样 64 routed + 2 shared 专家，K=6；200B tokens。
- 词表 32K；其余优化器/学习率沿用 DeepSeek-LLM 系列默认。
- **没有在更大规模（>3B）公开实验**；后续的真实大规模验证出现在 DeepSeek-V3 的 671B 训练中。

## 训练 & 系统细节

- 实现极轻量：无需新增可学参数，只多一个 batch 级别的统计与一行偏置更新，与现有 MoE infra 完全兼容。
- 关键工程点：偏置必须**用 FP32 存储与更新**（论文未明说，但 V3 实现按此做），避免 BF16 精度引发的累积漂移。
- 偏置在 inference 期间**不参与计算**（推理时是否仍加偏置取决于实现；V3 实现保留 b 用于推理 routing，确保和训练一致）。

## 对 16B MoE 设计的启示

1. **直接采用 Loss-Free 替代 expert/device/communication 三件套**：对 16B-class MoE，去除 α_1=0.003 的 expert-balance loss、α_2/α_3 的 device/comm loss，全部由 b_i + u=0.001 接管。
2. **保留一个极小的 sequence-wise complementary loss**（论文未提，但 V3 实操中用 α=1e-4 的 Eq. 17 形式，见论文 4），防止单条很长序列内部出现极端偏斜。
3. **u=1e-3 + sign 是默认值**；如果 16B 训练量很大（>2T tokens），可考虑训练末段把 u 衰减到 0（V3 即在最后 0.5T tokens 把 γ 调为 0），让模型不再被偏置干扰、回到纯 affinity routing。
4. **不要把偏置乘进 g_{i,t}**——这是论文反复强调的关键，破坏这一点等于退化为旧 aux-loss。
5. **MaxVio 是工程友好的监控指标**，应同时记录 MaxVio_batch / MaxVio_global，方便定位训练中偶发的"路由塌缩"。

## Caveats / 局限

- 论文最大实验仅到 3B / 200B tokens；其在 >100B 参数、>10T tokens 的稳定性证据来自 DeepSeek-V3 的工程报告（论文 4），不是本论文。
- 没有给出**长序列**或 **expert-parallel 多 device 设置**下 b_i 收敛性的理论分析；本质上 b_i 是 P-controller / integrator，没有形式化证明。
- 偏置的"sign 更新"在某些极端 batch（例如某专家在该 batch 完全无 token）下会一直增长，论文未讨论上界；实际工程上需要 clipping。
- 推理时是否仍带 b_i 影响 routing：论文没明确（"is not added to the g_{i,t}"），V3 实操是**保留 b_i 用于 top-K** 以保持 train/serve 一致。
- 文中没有对比 Hash Layer / Random Routing / Z-Loss 等"无需 balance"的路由方案；只对比了 Loss-Controlled。
- 这套方法解决"负载均衡"，但不解决"模式坍缩 / 专家死亡"，仍需要 sufficient warmup 与 sensible 初始化（如 V3 使用的随机 routing warmup）。
