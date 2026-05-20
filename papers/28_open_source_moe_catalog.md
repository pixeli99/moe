# 开源 MoE 架构全景统计（截至 2026-05）

> **目的**：把市面上所有公开权重 / 公开技术报告的 MoE LLM 架构维度按"可比 spec"形式整理成一张表，便于在 16B 设计决策中横向对照。
> **覆盖范围**：**62 个独立架构** + 6 个 paper-only 实验锚点（mHC 3B/9B/27B、DeepSeekMoE 2B、OLMoE 0.5B 等）。2024-01 ~ 2026-05 之间的开源 MoE LLM 基本全覆盖；纯 dense 模型（Mistral Large 系、Tele-FLM、Aya、Apriel、Skywork-OR1）不入表。
> **数据来源**：HuggingFace `config.json` 优先，论文表格其次，blog 第三。空值标 ?；标注列指出与 V3 / Mixtral / OLMoE 派系差异点。

---

## 1. TL;DR（六条核心结论）

1. **激活率（active params / total params）收敛到 1/15 ± 1/8**：⚠️ **本表所有"激活率"统一为 active/total params 主口径**（不是 expert-slot fraction）。极稀疏端 Qwen3-Next (3.8%, 1/26) / LongCat (4.8%, 1/21) / K2 (3.2%, 1/31) 与极稠密端 OLMoE / JetMoE (1/4-1/5) 之间，**主流卡在 1/12 ~ 1/16**（V3=18×, Hunyuan-Large=7.5×, Mixtral=6×, Qwen3-30B=10×）。注意 "expert-slot fraction" 是另一口径：Ling 全系 (8+1)/(256+1) ≈ 3.5%，dots1 (6+2)/(128+2) ≈ 6.2% —— 这跟 active/total 数值上不同，引用时必须标明。
2. **Top-K 标准化到 8**：2024 主流是 2（Mixtral / DBRX / Grok-1 / Phi-3.5 / Switch top-1）；2025 之后**新模型几乎全 K=8**（V3, Ling 全系, GLM-4.5, dots1, ERNIE 4.5, MiniMax-M2；gpt-oss=4 是少数例外）。**Hunyuan-Large 与 Llama 4 走 K=1** 是反例（注意 Hunyuan-Large 是 K=1 routed + 1 shared 而不是 K=8）。
3. **Sigmoid + ALF 已成 2025+ 默认路由**：DeepSeek-V3 起的 `sigmoid gate + Aux-Loss-Free bias` 范式被 Ling 全系 / GLM-4.5 / Moonlight / dots1（部分）/ Qwen3-Coder 继承；老派 `softmax + aux-loss` 仍是 Mixtral / Qwen3-Dense 系 / Llama 4 / gpt-oss 的选择。**两个生态目前各占一半，但增长在 sigmoid 一侧**。
4. **共享专家是"国产派 vs 西方派"分水岭**：DeepSeek 系 / Qwen 系（除 Qwen3-Coder/235B-2507）/ Ling / Hunyuan / Pangu / GLM 全用 1 个 shared expert；Mixtral / DBRX / OLMoE / gpt-oss / Grok 全不用。Llama 4 用"shared MLP + routed 1 expert"是特殊形态。
5. **MTP 与 hybrid attention 是 2025+ 两条上行曲线**：MTP 从 V3 (1 chain) 扩散到 GLM-4.5 / Ling-1T / Ring-1T / Qwen3-Next / MiniMax-M2 (D=3)；hybrid attention（Mamba/SSM 或 Linear+softmax 交替）从 Jamba / BlackMamba 扩散到 MiniMax-01/M1 / Granite 4 / Nemotron-3 Nano / Qwen3-Next / Hunyuan-Turbo-S。
6. **超大与超小两头开花**：1T 级 (Kimi K2 / Ling-1T / Ring-1T / Intern-S1-Pro / GLaM) + 80B-100B 级 (Llama 4 Scout / Qwen3-Next / Ling-flash / Hunyuan-A13B / Hy3) + 16-30B 级 (V2-Lite / Ling-mini / DeepSeekMoE-16B / OLMoE / gpt-oss-20b / Phi-3.5-MoE / Granite 3.x / Qwen3-30B-A3B / dots-MoE-mini / Moonlight) 三个明显聚类。**16B 总参带是 2024-Q2 ~ 2026-Q1 增长最快的稀疏量级**。

---

## 2. 主表（按总参数升序）

> 缩写：**Attn** = MHA(普通) / GQA / MLA(DeepSeek 多潜在头) / Hybrid(混合 SSM/Linear)；**Route** = softmax / sigmoid / softmax+aux 等；**Bal** = aux-loss / ALF (aux-loss-free) / GRIN / Sinkhorn / none；**MTP** = D 数 / no。

| # | Model | Org | Date | Total | Active | Sp. | L | Hid | N\_rt | K | N\_sh | E\_FFN | Attn | Q/KV | Vocab | Route | Bal | MTP | 训 tok | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **MoE-LLaVA-3B** | PKU | 24-01 | 3B | ~2B | 0.67 | 24 | 2048 | 4 | 2 | 0 | ? | MHA | 16/16 | 32K | softmax | loss | – | – | 多模态 |
| 2 | **Granite 3.x 1B-A400M** | IBM | 24-10 | 1.3B | 400M | 0.31 | 24 | 1024 | 32 | 8 | 0 | 512 | GQA | 16/8 | 49152 | softmax | loss | – | 10T | tied emb |
| 3 | **Granite 3.x 3B-A800M** | IBM | 24-10 | 3.4B | 800M | 0.24 | 32 | 1536 | 40 | 8 | 0 | 512 | GQA | 24/8 | 49152 | softmax | loss | – | 12T | scale-mult / dropless |
| 4 | **BlackMamba 2.8B** | Zyphra | 24-02 | 2.8B | ~0.6B | 0.21 | 36 | 1472 | 8 | 1 | 0 | 3872 | **Mamba SSM** | – | 50304 | Sinkhorn | Sinkhorn | – | 300B | 纯 SSM + MoE |
| 5 | **MiniCPM-MoE-8x2B** | OpenBMB | 24-04 | 13.6B | ~4B | 0.29 | 40 | 2304 | 8 | 2 | 0 | 5760 | MHA | 36/36 | 122753 | softmax | loss | – | ? | scale\_depth=1.4 |
| 6 | **Qwen1.5-MoE-A2.7B** | Alibaba | 24-03 | 14.3B | 2.7B | 0.19 | 24 | 2048 | 60 | 4 | 4 | 1408 | MHA | 16/16 | 151936 | softmax | loss | – | ? | upcycled from Qwen-1.8B |
| 7 | **JetMoE-8B** | MIT-IBM | 24-04 | 8B | 2.2B | 0.27 | 24 | 2048 | 8 | 2 | 0 | 5632 | GQA + MoA(attn experts) | 32/16 | 32K | softmax | loss | – | 1.25T | attention 也 MoE |
| 8 | **DeepSeekMoE-16B** | DeepSeek | 24-01 | 16.4B | 2.8B | 0.17 | 28 | 2048 | 64 | 6 | 2 | 1408 | MHA | 16/16 | 100K | softmax | loss + aux\_α=0.001 | – | 2T | 现代 MoE 范式起点 |
| 9 | **DeepSeek-V2-Lite** | DeepSeek | 24-05 | 15.7B | 2.4B | 0.15 | 27 | 2048 | 64 | 6 | 2 | 1408 | **MLA** | 16/4(KV-lora) | 102400 | softmax | loss | – | 5.7T | 16B 设计 baseline |
| 10 | **Ling-lite (1.0)** | InclusionAI | 24-12 | 16.8B | 2.75B | 0.16 | 28 | 2048 | 64 | 6 | 2 | 1408 | MHA | 16/4 | 126464 | softmax | loss | – | ? | Ling 1.0 generation |
| 11 | **Moonlight-16B-A3B** | Moonshot+UCLA | 25-02 | 15.3B | 2.24B | 0.15 | 27 | 2048 | 64 | 6 | 2 | 1408 | MHA | 16/16 | 163840 | **sigmoid** | **ALF (noaux\_tc)** | – | 5.7T | V3 配方 + Muon 优化器 |
| 12 | **Ling-mini-2.0** | InclusionAI(Ant) | 25-09 | 16B | **1.4B** | **0.088** | 20 | 2048 | **256** | 8 | 1 | 512 | GQA + QK-norm + part-RoPE | 16/4 | 157184 | sigmoid+group | ALF | 1 | 20T+ | 1/32 稀疏极致；n\_group=8 topk\_group=4 |
| 13 | **gpt-oss-20b** | OpenAI | 25-08 | 21B | 3.6B | 0.17 | 24 | 2880 | 32 | 4 | 0 | 2880 | GQA + sliding/full alt | 64/8 | 201088 | softmax | loss (β=0.9) | – | ? | MXFP4；attn bias |
| 14 | **ERNIE 4.5 21B-A3B** | Baidu | 25-06 | 21B | 3B | 0.14 | 28 | 2560 | 64 | 6 | 2 | 1536 | GQA | 20/4 | 103424 | softmax | loss | – | ? | tied emb |
| 15 | **XVERSE-MoE-A4.2B** | XVERSE | 24-04 | 25.8B | 4.2B | 0.16 | 28 | 2560 | 64 | 6 | 2 | 1728 | MHA | 32/32 | 100534 | softmax | loss (0.01) | – | 3.2T | 8K ctx |
| 16 | **Aria** | Rhymes AI | 24-10 | 25.3B | 3.5B | 0.14 | 28 | 2560 | 64 | 6 | 2 | 1664 | MHA | 20/20 | 100352 | softmax | loss | – | 6.4T | 多模态原生 MoE，64K ctx |
| 17 | **Qwen3-30B-A3B** | Alibaba | 25-04 | 30B | 3B | 0.10 | 48 | 2048 | 128 | 8 | 0 | 768 | GQA + QK-norm | 32/4 | 151936 | softmax | loss | – | 36T | RoPE 1e7 |
| 18 | **Granite 4.0 H-Small** | IBM | 25-10 | 32B | 9B | 0.28 | hybrid | ? | ? | ? | 1 (always-on) | ? | **Hybrid Mamba-2+Attn** | ? | ? | softmax | loss | – | ? | "always-on" shared expert |
| 19 | **Yuan2.0-M32** | IEIT | 24-05 | 40B | 3.7B | 0.092 | 24 | 2048 | 32 | 2 | 0 | 8192 | LFA | 16/16 | 135040 | **Attention Router** | loss + maxz | – | 2T | 唯一用 attention router |
| 20 | **Phi-3.5-MoE** | Microsoft | 24-08 | 41.9B | 6.6B | 0.16 | 32 | 4096 | 16 | 2 | 0 | 6400 | GQA | 32/8 | 32064 | softmax | **GRIN** | – | 4.9T | GRIN gradient-routing |
| 21 | **Jamba-v0.1** | AI21 | 24-03 | 52B | 12B | 0.23 | 32 | 4096 | 16 | 2 | 0 | 14336 | **Mamba+Attn 7:1** | 32/8 | 65536 | softmax | loss (1e-3) | – | ? | hybrid 鼻祖 |
| 22 | **Jamba-1.5-Mini** | AI21 | 24-08 | 52B | 12B | 0.23 | 32 | 4096 | 16 | 2 | 0 | 14336 | Mamba+Attn 7:1 | 32/8 | 65536 | softmax | loss | – | ? | 256K ctx |
| 23 | **Mixtral 8x7B** | Mistral | 23-12 | 46.7B | 12.9B | 0.28 | 32 | 4096 | 8 | 2 | 0 | 14336 | GQA | 32/8 | 32K | softmax | loss | – | ? | 业界 1.0 标杆 |
| 24 | **Qwen2-57B-A14B** | Alibaba | 24-06 | 57B | 14B | 0.25 | 28 | 3584 | 64 | 8 | **8 (shared)** | 2560 | GQA | 28/4 | 151936 | softmax | loss | – | 7T | shared\_FFN=20480 |
| 25 | **Pangu Pro MoE** | Huawei | 25-05 | 72B | 16B | 0.22 | 48 | 5120 | 64 | 8 (**1/group**) | 1 | 1344 | GQA | 40/8 | 153376 | **MoGE (grouped)** | loss | – | 13T | 唯一 grouped routing |
| 26 | **Qwen3-Next-80B-A3B** | Alibaba | 25-09 | 80B | **3B** | **0.038** | 48 | 2048 | **512** | 10 | 1 | 512 | **Hybrid DeltaNet+Attn 3:1** | 16/2 | 151936 | softmax | loss (1e-3) | yes | ? | 1/32 sparsity; head\_dim=256 |
| 27 | **Hunyuan-A13B** | Tencent | 25-06 | 80B | 13B | 0.16 | 32 | 4096 | 64 | 8 | 1 | 3072 | GQA + QK-norm | 32/8 | 128167 | softmax | loss | – | ? | dynamic RoPE, 256K |
| 28 | **GLM-4.5-Air** | Z.ai | 25-07 | 106B | 12B | 0.11 | 46 | 4096 | 128 | 8 | 1 | 1408 | GQA + part-RoPE | 96/8 | 151552 | **sigmoid** | none (ALF) | – | ? | first\_k\_dense=1 |
| 29 | **Llama 4 Scout** | Meta | 25-04 | 109B | 17B | 0.156 | 48 | 5120 | 16 | **1** | 1 (shared MLP 16384) | 8192 | GQA + **iRoPE** | 40/8 | 202048 | softmax | loss | – | ~40T | K=1 + 10M ctx |
| 30 | **Ling-flash-2.0** | InclusionAI | 25-09 | 100B | 6.1B | 0.061 | 32 | 4096 | 256 | 8 | 1 | 1024 | GQA + QK-norm | 32/4 | 157184 | sigmoid | ALF | yes | 20T+ | 1/32 稀疏 |
| 31 | **Ring-flash-2.0** | InclusionAI | 25-10 | 100B | 6.1B | 0.061 | 32 | 4096 | 256 | 8 | 1 | 1024 | GQA + QK-norm | 32/4 | 157184 | sigmoid | ALF | yes | ? | reasoning variant |
| 32 | **gpt-oss-120b** | OpenAI | 25-08 | 117B | 5.1B | 0.044 | 36 | 2880 | 128 | 4 | 0 | 2880 | GQA + sliding/full alt | 64/8 | 201088 | softmax | loss (β=0.9) | – | ? | YaRN→128K；head\_dim=**64** |
| 33 | **AquilaMoE 8x16B** | BAAI | 24-08 | 128B | ~30B | 0.23 | 40 | 5120 | 8 | 2 | 0 | 20480 | GQA | 40/8 | 151851 | softmax | loss | – | ? | EfficientScale 上 cycle |
| 34 | **DBRX** | Databricks | 24-03 | 132B | 36B | 0.27 | 40 | 6144 | 16 | 4 | 0 | 10752 | GQA | 48/8 | 100352 | softmax | loss | – | 12T | RoPE=5e5 |
| 35 | **WizardLM-2-8x22B** | MS-WizardLM | 24-04 | 141B | 39B | 0.28 | 56 | 6144 | 8 | 2 | 0 | 16384 | GQA | 48/8 | 32K | softmax | loss | – | (FT) | Mixtral-8x22B 后训 |
| 36 | **dots.llm1** | rednote | 25-06 | 142B | 14B | 0.099 | 62 | 4096 | 128 | 6 | 2 | 1408 | MHA + **QK-Norm** | 32/32 | 152064 | softmax / sigmoid+norm | loss + ALF mix | – | 11.2T | routed\_scale=2.5 |
| 37 | **Mixtral 8x22B** | Mistral | 24-04 | 141B | 39B | 0.28 | 56 | 6144 | 8 | 2 | 0 | 16384 | GQA | 48/8 | 32K | softmax | loss | – | ? | Mixtral 系顶配 |
| 38 | **MiniMax-Text-01** | MiniMax | 25-01 | 456B | 45.9B | 0.10 | 80 | 6144 | 32 | 2 | 0 | 9216 | **Lightning Attn (linear) + softmax 1/8** | 64/8 | 200K | softmax | loss | – | ? | 10M ctx |
| 39 | **MiniMax-M1** | MiniMax | 25-06 | 456B | 45.9B | 0.10 | 80 | 6144 | 32 | 2 | 0 | 9216 | Lightning 1:7 | 64/8 | 200K | softmax | loss | – | +7.5T | reasoning, 1M ctx |
| 40 | **MiniMax-M2** | MiniMax | 25-10 | 230B | **10B** | 0.043 | 62 | 3072 | 256 | 8 | 0 | 8192 | GQA + QK-norm（**抛弃 lightning**） | 48/8 | 200K | softmax | loss (1e-3) | **D=3** | ? | FP8 e4m3, agentic 取向 |
| 41 | **Hunyuan-Large** | Tencent | 24-11 | 389B | 52B | 0.13 | 64 | 6400 | 16 | 1 | 1 | ? | GQA + CLA + QK-norm | 80/8 | 129024 | softmax | loss | – | 7T | recycle routing + 1.5T synthetic |
| 42 | **Jamba-1.5-Large** | AI21 | 24-08 | 398B | 94B | 0.24 | 72 | 8192 | 16 | 2 | 0 | ? | Mamba+Attn 1:7 | 64/8 | 65536 | softmax | loss | – | ? | ExpertsInt8 |
| 43 | **Llama 4 Maverick** | Meta | 25-04 | 400B | 17B | 0.043 | 48 | 5120 | 128 | **1** | 1 (shared) | 8192 | GQA + iRoPE | 40/8 | 202048 | softmax | loss | – | ~22T | K=1 + 1M ctx |
| 44 | **Snowflake Arctic** | Snowflake | 24-04 | 480B | 17B | 0.035 | 35 | 7168 | 128 | 2 | **dense residual** | 4864 | GQA | 56/8 | 32K | softmax | loss | – | 3.5T | 10B dense + MoE residual |
| 45 | **Qwen3-Coder-480B-A35B** | Alibaba | 25-07 | 480B | 35B | 0.073 | 62 | 6144 | 160 | 8 | 0 | 2560 | GQA + QK-norm | 96/8 | 151936 | softmax (norm-topk) | none | – | ? | 256K, YaRN→1M |
| 46 | **Step-3** | StepFun | 25 | 321B | 38B | 0.118 | 61 | 7168 | 48 | 3 | 1 | 5120 | MFA (Multi-Matrix Attn) | 64/16 | 128K | softmax | loss + balanced | – | ? | MFA 新注意力 |
| 47 | **ERNIE 4.5 300B-A47B** | Baidu | 25-06 | 300B | 47B | 0.157 | 54 | 8192 | 64 | 8 | 0 | 3584 | GQA | 64/8 | 103424 | softmax | loss (1e-3) | – | ? | MoE 仅 layer 3-53 |
| 48 | **GLM-4.5** | Z.ai | 25-07 | 355B | 32B | 0.090 | **92** | 5120 | 160 | 8 | 1 | 1536 | GQA + part-RoPE + QK-norm | 96/8 | 151552 | **sigmoid** | none (ALF) | **D=1** | 23T | 深 > 宽，routed\_scale=2.5 |
| 49 | **GLM-4.6** | Z.ai | 25-09 | 355B | 32B | 0.090 | 92 | 5120 | 160 | 8 | 1 | 1536 | GQA + part-RoPE + QK-norm | 96/8 | 151552 | sigmoid | none | D=1 | ? | 200K, 同 4.5 arch |
| 50 | **DeepSeek-V2-236B** | DeepSeek | 24-05 | 236B | 21B | 0.089 | 60 | 5120 | 160 | 6 | 2 | 1536 | **MLA** | 128/kv-lora 512 | 100K | softmax | loss + device-limited | – | 8.1T | MLA 起点 |
| 51 | **Ling-plus** | InclusionAI | 24-12 | 290B | 28.8B | 0.099 | 88 | 5376 | 64 | 4 | 1 | 3072 | GQA | 56/8 | 126464 | softmax | loss | – | ? | Ling 1.0 顶配 |
| 52 | **Grok-1** | xAI | 24-03 | 314B | ~79B | 0.25 | 64 | 6144 | 8 | 2 | 0 | ? | MHA(类GQA) | 48/8 | 131072 | softmax | loss | – | ? | Apache 2.0, 8K ctx |
| 53 | **Skywork-MoE** | Skywork | 24-06 | 146B | 22B | 0.151 | 52 | 4608 | 16 | 2 | 0 | ? | GQA | 36/4 | 151936 | softmax | **TwoStage gating + GLU experts** | – | 0.55T | upcycled, gate logit norm |
| 54 | **Hy3 (Hunyuan-3.0 preview)** | Tencent | 25-09 | 295B | 21B | 0.071 | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | 256K, weights gated |
| 55 | **Hunyuan-Turbo-S** | Tencent | 25-03 | 560B | 56B | 0.10 | 128 | ? | 32 | 2 | 1 | ? | **Hybrid Attn+Mamba2** | ? | ? | softmax | loss | – | 16T | 首个 Mamba MoE @ scale |
| 56 | **LongCat-Flash** | Meituan | 25-09 | ~560B | ~27B | 0.048 | 28 | 6144 | 512 | 12 | **256 zero-experts** | 2048 | MLA | 64/kv-lora 512 | 131072 | softmax | loss | – | ? | "零专家"恒等映射 + routed\_scale=6 |
| 57 | **DeepSeek-V3** | DeepSeek | 24-12 | 671B | 37B | 0.055 | 61 | 7168 | 256 | 8 | 1 | 2048 | MLA | 128/kv-lora 512 | 129280 | **sigmoid** | **ALF (noaux\_tc)** | **D=1** | 14.8T | ALF + sigmoid 范式起点 |
| 58 | **DeepSeek-V3.1** | DeepSeek | 25-08 | 671B | 37B | 0.055 | 61 | 7168 | 256 | 8 | 1 | 2048 | MLA | 128/kv-lora 512 | 129280 | sigmoid | ALF | D=1 | +840B | hybrid thinking + UE8M0 FP8 |
| 59 | **DeepSeek-V3.2-Exp** | DeepSeek | 25-09 | 671B | 37B | 0.055 | 61 | 7168 | 256 | 8 | 1 | 2048 | **DSA (sparse attn over MLA)** | 128/kv-lora 512 | 129280 | sigmoid | ALF | D=1 | ? | sparse attention 实验 |
| 60 | **Nemotron-3 Nano 30B-A3B** | NVIDIA | 25-12 | 31.6B | 3.2B | 0.10 | 29 hybrid | ? | 128 | 5–6 | 1 | ? | Hybrid Mamba-2+GQA | ? | ? | softmax | loss | – | ? | 1M ctx |
| 61 | **Kimi K2** | Moonshot | 25 | 1T | 32B | 0.032 | 61 | 7168 | 384 | 8 | 1 | 2048 | MLA + **MuonClip** | **64**/kv-lora 512 | 163840 | sigmoid | ALF | – | 15.5T | 1T 级首发；Muon 优化器 + post-QK-Clip |
| 62 | **Ling-1T** | InclusionAI | 25-10 | 1T | ~50B | 0.05 | 80 | 8192 | 256 | 8 | 1 | 2048 | GQA + QK-norm | 64/8 | 157184 | sigmoid | ALF | D=1 | 20T+ | 最大 FP8 base 模型 |
| 63 | **Ring-1T** | InclusionAI | 25-10 | 1T | ~50B | 0.05 | 80 | 8192 | 256 | 8 | 1 | 2048 | GQA + QK-norm | 64/8 | 157184 | sigmoid | ALF | D=1 | 20T+ | Ling-1T 上 RL（icepop） |
| 64 | **Intern-S1-Pro** | Shanghai AI Lab | 26-02 | 1T | 22B | 0.022 | ? | ? | 512 | 8 | ? | ? | ? | ? | ? | ? | ? | ? | ? | 科学多模态 |
| 65 | **GLaM** | Google | 21-12 | 1.2T | 96.6B | 0.08 | 32 MoE | 8192 | 64 | 2 | 0 | ? | dec-only | ? | 256K | softmax | loss | – | 1.6T | 历史性，权重未开 |
| 66 | **Switch Transformer XXL** | Google | 21-01 | 1.6T | ~ | – | 24 | ? | **2048** | **1** | 0 | ? | T5 enc-dec | ? | T5 vocab | softmax | loss | – | 503B | top-1 路由首发 |

### Paper-only 实验锚点（不是独立 release，用作 wind tunnel 对照）

| Model | Paper | Total | Active | Layers | Hidden | N\_rt | K | N\_sh | E\_FFN | Attn | 训 tok | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mHC 3B（短）** | 2512.24880 | 2.97B | 612M | 12 | 1280 | 64 | 6 | 2 | 896 | MLA | 39.3B | 本仓库 A2 锚点直接对标 |
| **mHC 3B（长）** | 2512.24880 | 2.97B | 612M | 12 | 1280 | 64 | 6 | 2 | 896 | MLA | 1.05T | 同上但 1T tokens |
| **mHC 9B** | 2512.24880 | 9.18B | 1.66B | 18 | 1920 | 64 | 6 | 2 | 1280 | MLA | 105B | – |
| **mHC 27B（主）** | 2512.24880 | 27.0B | 4.14B | 30 | 2560 | 72 | 6 | 2 | 1536 | MLA | 262B | HC vs mHC 稳定性主对比 |
| **DeepSeekMoE 2B** | 2401.06066 | 2.0B | 0.31B | 9 | 1280 | 64 | 6 | 1 | 896 | MHA | 100B | DeepSeek 1.0 anchor |
| **OLMoE-1B-7B** | 2409.02060 | 6.9B | 1.3B | 16 | 2048 | 64 | 8 | 0 | 1024 | MHA | 5T | 完全开源 corpus+ckpt |
| **AttnRes anchors** | 2603.15031 | 0.5–3.5B | 194-528M | 多档 | – | 64 | 8 | 0/1 | – | – | – | Kimi 团队的 scaling-law 锚点 |
| **GShard** | 2006.16668 | 600B | – | enc-dec | – | 2048 | 2 | – | – | – | – | top-2 路由原型 |

---

## 3. 按维度的分布统计

### 3.1 激活率（active / total，独立 release 模型，共 66 个 distinct architectures）

| 激活率区间 | 个数 | 代表 |
|---|---|---|
| **≥ 25%（半稠密）** | 9 | DBRX, Mixtral 系, Grok-1, Qwen2-57B, Jamba 1.5 Mini, WizardLM-2, AquilaMoE |
| **15–25%（标准 MoE）** | 19 | DeepSeekMoE-16B, V2-Lite, Ling-mini-1.0, Phi-3.5, Hunyuan-Large, Llama 4 Scout, ERNIE 4.5 300B, Skywork, OLMoE, Moonlight, gpt-oss-20b, JetMoE, ... |
| **8–15%（V3 主流）** | 24 | V3 全系 (5.5%→重分), **Ling-mini-2.0 (8.8%)**, Qwen3-30B/235B, Ling-flash (5.9%→重分), Ring-flash, GLM-4.5/4.6/Air (9%, 11.3%), Step-3, dots1 **(9.9%)**, Hunyuan-A13B, Nemotron-3 Nano, MiniMax-01/M1, Pangu Pro ... |
| **< 8%（极稀疏）** | 14 | K2 (3.2%), GLaM (8%), Llama 4 Maverick (4.3%), gpt-oss-120b (4.4%), MiniMax-M2 (4.3%), Snowflake Arctic (3.5%), Qwen3-Next (3.8%), LongCat (4.8%), Ring/Ling-1T (5.1%), Ling-flash-2.0 (5.9%), Intern-S1-Pro (2.2%) |

**关键趋势**：2025-H2 之后新模型几乎全部进入 < 10% 区间。"超稀疏 + 大量 expert" 是新主流。

### 3.2 Top-K 分布

| Top-K | 个数 | 代表 |
|---|---|---|
| K=1 | 5 | **Llama 4 Scout / Maverick**, Switch Transformer, BlackMamba, **Hunyuan-Large** (K=1 routed + 1 shared) |
| K=2 | 14 | Mixtral 全系, Snowflake Arctic, Phi-3.5, Yuan-M32, Jamba 全系, MiniMax-01/M1, Hunyuan-Turbo-S, OpenMoE, GLaM ... |
| K=3 | 1 | Step-3 |
| K=4 | 3 | DBRX, **gpt-oss 120b/20b**, Qwen1.5-MoE |
| K=6 | 11 | DeepSeekMoE-16B, V2-Lite, V2-236B (K=6), Moonlight, ERNIE-21B-A3B, Aria, XVERSE, Ling-lite, mHC anchors, ... |
| **K=8** | **23** | **V3/V3.1/V3.2, K2, Hunyuan-A13B, Ling 2.0 全系, GLM-4.5/4.6/Air, dots1, ERNIE 300B, Qwen3-30B/235B/Coder/VL, Qwen2-57B, Granite 3.x, gpt-oss(4 例外), Pangu Pro, MiniMax-M2, Nemotron-3 Nano, Intern-S1-Pro** |
| K=10 | 1 | Qwen3-Next-80B |
| K=12 | 1 | LongCat-Flash |

→ **K=8 是 2025+ 默认值**；2024 一代主流 K=2。

### 3.3 共享专家（N_shared）

| 配置 | 个数 | 代表 |
|---|---|---|
| **0 个 shared** | 24 | Mixtral 全系, DBRX, OLMoE, Grok-1, Snowflake (用 dense residual 不计), gpt-oss 全系, Llama 4（用 shared MLP 不计入 N\_shared）, Switch, Qwen3-30B, Qwen3-Coder, Yuan-M32, MiniMax 全系 ... |
| **1 个 shared** | 24 | V3 全系, K2, Hunyuan-Large, Ling-2.0 全系, GLM-4.5/4.6/Air, Step-3, Hunyuan-A13B, Pangu Pro, Llama 4 (1 shared MLP), Granite 4, LongCat 等 |
| **2 个 shared** | 12 | DeepSeekMoE-16B, V2-Lite, V2-236B, Moonlight, ERNIE-21B-A3B, Aria, XVERSE, Ling-1.0 lite/plus, dots1, mHC anchors（3B/9B/27B）, MiniCPM-MoE |
| **多个 shared / shared FFN** | 3 | Qwen1.5-MoE（4 个 shared）, Qwen2-57B-A14B（8 个 shared）, Snowflake（dense FFN residual） |

→ **DeepSeek 阵营从 V2 的 2 shared 在 V3 砍到 1 shared**；Ling 2.0 / Moonlight 跟进 1；Qwen MoE 在 1.5/2 用过 multi-shared，3 代弃用 shared（除 Coder）。

### 3.4 路由函数与 balance 策略（仅独立 release 模型）

| 路由 + balance | 个数 | 代表 |
|---|---|---|
| **softmax + aux-loss** | 35 | Mixtral, DBRX, OLMoE, Skywork, Yuan-M32, Hunyuan-Large/A13B, MiniMax 全系, gpt-oss, Llama 4, ERNIE 4.5 全系, Pangu Pro, Qwen3-30B/235B/Coder, Jamba 全系, JetMoE, MiniCPM-MoE, Phi-3.5(GRIN), ... |
| **sigmoid + ALF (Aux-Loss-Free, V3 风格)** | 14 | **V3/V3.1/V3.2, K2, Moonlight, Ling-mini/flash-2.0 + 1T, Ring 全系, GLM-4.5/4.5-Air/4.6, dots1（变体）** |
| **特殊路由器** | 5 | Yuan-M32（attention router）, BlackMamba（Sinkhorn）, Pangu Pro（MoGE grouped）, Phi-3.5（GRIN gradient routing）, Switch（top-1 + load balance loss） |

→ **sigmoid+ALF 占新模型增量主导**。注：dots1 论文里说沿用 Ling 配方但保留 aux-loss，且 routed\_scaling\_factor=2.5 = V3 配方指纹。

### 3.5 Attention 形态

| 类型 | 个数 | 代表 |
|---|---|---|
| **MHA** | 11 | DeepSeekMoE-16B, Moonlight, Qwen1.5-MoE, dots1, XVERSE, Aria, OpenMoE, MoE-LLaVA, MiniCPM-MoE, Yuan-M32(LFA), Grok-1 |
| **GQA** | 26 | Mixtral 全系, DBRX, Snowflake, Phi-3.5, Qwen2-57B, Qwen3 全系（30B/235B/Coder/VL）, Hunyuan-A13B, Hunyuan-Large, Llama 4 系, GLM-4.5 全系, ERNIE 全系, Skywork, AquilaMoE, gpt-oss 全系, Pangu Pro, Granite 3.x, Ling-flash/1T, Ring-flash/1T, JetMoE ... |
| **MLA**（DeepSeek） | 8 | V2/V2-Lite/V3/V3.1/V3.2-Exp, K2, mHC anchors, LongCat-Flash |
| **MFA / 其他** | 1 | Step-3 (Multi-Matrix Attn) |
| **Hybrid (Mamba/SSM/Linear)** | 9 | Jamba 全系（3 个）, BlackMamba, MiniMax-01/M1, Granite 4 全系, Nemotron-3 Nano 全系, Qwen3-Next, Hunyuan-Turbo-S |

→ **GQA 是绝对主流**；MLA 是 DeepSeek 阵营专属；hybrid attention 在 long-context / 长生成场景兴起。

### 3.6 MTP（Multi-Token Prediction）

| 配置 | 个数 | 代表 |
|---|---|---|
| **不用** | 50+ | 大多数 2024 模型 + Qwen3-30B/235B-2507 + Hunyuan-A13B/Large + Phi-3.5 + Mixtral 全系 + DBRX + gpt-oss + Llama 4 全系 + Granite 全系 + JetMoE + Skywork + dots1 + ... |
| **D=1（V3 风格 causal chain）** | 9 | V3/V3.1/V3.2, GLM-4.5/4.6, Ling-mini/flash/1T-2.0, Ring 全系, Qwen3-Next |
| **D=3（多层 chain）** | 1 | **MiniMax-M2** |

→ MTP 在 sigmoid+ALF 阵营中 ~70% 采用；softmax 阵营 ~0%。两者强相关。

### 3.7 训练 tokens 量（仅有数据的模型）

| 区间 | 代表 |
|---|---|
| **< 1T** | Switch (503B), GLaM (1.6T), Skywork-MoE (550B), OpenMoE-34B (200B), BlackMamba (300B) |
| **1–5T** | DeepSeekMoE-16B (2T), Yuan-M32 (2T), Llama 4 Behemoth (?), AquilaMoE, XVERSE (3.2T), Snowflake Arctic (3.5T) |
| **5–10T** | V2-Lite (5.7T), Moonlight (5.7T), Aria (6.4T), Hunyuan-Large (7T), Qwen2-57B (7T), V2-236B (8.1T) |
| **10–15T** | DBRX (12T), Granite 3.x 3B (12T), Pangu Pro (13T), V3 (14.8T) |
| **15–25T** | K2 (15.5T), Hunyuan-Turbo-S (16T), Ling 2.0 全系 (20T+), Ring 全系 (20T+), Ling-1T (20T+), GLM-4.5 (23T), Llama 4 Maverick (~22T) |
| **30T+** | Qwen3-30B/235B (~36T), Llama 4 Scout (~40T) |

→ 增长趋势明显：**2024 模型在 2-12T，2025-H2 之后 20-40T 成为新标准**。

---

## 4. 七大典型 design pattern

### Pattern A：DeepSeek-V3 派（sigmoid + ALF + MLA + 1 shared + MTP D=1）
**成员**：V3 / V3.1 / V3.2-Exp / K2 / Moonlight / Ling-2.0 全系 / Ring 全系 / GLM-4.5/4.5-Air/4.6 / LongCat-Flash（部分）
**共性**：sigmoid gate + `routed_scaling_factor` (V3=2.5, LongCat=6)、bias-based ALF（Wang et al. 2408.15664）、1 shared expert、K=8、MTP D=1 训练辅助
**变体差异**：
- K2 用 MuonClip post-QK-Clip 而不是 QK-Norm
- Ling 2.0 引入零均值 ALF bias 更新（论文 Eq. 6-7）
- GLM-4.5 用 partial RoPE（仅部分 head 加 rotary）+ depth>width 路线
- LongCat 加入"零专家"（identity routing）

### Pattern B：Mixtral 派（softmax + aux-loss + GQA + 0 shared + K=2）
**成员**：Mixtral 8x7B / 8x22B / WizardLM-2 8x22B / DBRX (K=4 是小修正) / Phi-3.5-MoE / AquilaMoE / Grok-1（K=2）
**共性**：稠密激活率 25-30%、少量 expert (8-16)、K=2、无 shared、softmax routing + standard load-balancing loss
**问题**：稠密激活率下 inference compute 高，被 V3 派以"更稀疏 + 更多 expert"取代

### Pattern C：OLMoE/JetMoE 派（fine-grained + 8 experts active + 0 shared）
**成员**：OLMoE-1B-7B / JetMoE-8B / Granite 3.x / mHC 27B anchors（部分）
**共性**：K=8、active 在总参 15-30% 区间、强调"细粒度专家"、不用 shared
**特色**：OLMoE 是当前最完整的"全开源 corpus + ckpt"MoE；Granite 强调 dropless 路由

### Pattern D：Llama 4 派（K=1 + shared MLP + 大 expert）
**成员**：Llama 4 Scout / Maverick（Behemoth pending）
**共性**：K=1（单 expert + 共享 MLP，total active=2 个 FFN）、expert FFN 巨大（8192）、iRoPE
**风险**：K=1 历史上（Switch Transformer）下游 quality 弱于 K≥2；Maverick 1M ctx 是核心卖点

### Pattern E：MoE + 混合注意力派（Mamba/SSM/Linear 配 MoE）
**成员**：Jamba 全系（Mamba+Attn 7:1）/ BlackMamba（纯 Mamba+MoE）/ MiniMax-01/M1（Lightning Attn 1:7）/ Granite 4 全系（Mamba-2+Attn）/ Nemotron-3 Nano 全系（Mamba-2+GQA）/ Qwen3-Next（Gated DeltaNet+Attn 3:1）/ Hunyuan-Turbo-S
**共性**：用线性 / SSM 注意力替代大部分层，softmax 注意力保留少数层做 retrieval/precision；MoE 嵌在 FFN 部分
**特色**：MiniMax-M2 反向 — 抛弃 lightning 注意力回到 full softmax，说明 hybrid 还在探索中

### Pattern F：Switch / GLaM 历史派（top-1 / top-2 + 极大 N_routed）
**成员**：Switch Transformer (2048 experts, K=1) / GLaM (64 experts, K=2) / Snowflake Arctic (128 experts, K=2, 但有 dense residual)
**共性**：极端 expert 数量、top-1 或 top-2、稀疏度高 但 active 也大
**现状**：被现代 V3 派取代，仅 Snowflake 是 production-grade 后裔

### Pattern G：异构创新派
- **Pangu Pro MoGE**：把 N=64 routed 分 8 组，每组强制 top-1（共 top-8）—— grouped balance 替代 ALF
- **Yuan-M32**：用 attention router 而非 affinity score
- **Phi-3.5 GRIN**：gradient-routed gating，路由权重也走梯度
- **Skywork-MoE**：两阶段 gating + GLU experts + gate logit normalization
- **Step-3**：MFA (Multi-Matrix Attention) + K=3
- **DeepSeek-V3.2-Exp**：Sparse Attention (DSA) 替代 MLA 的 dense 部分
- **LongCat zero-experts**：256 个 identity routing slot，让 token 可以"跳过"专家

---

## 5. 16B 设计可借鉴的关键发现

> 用本统计验证 22\_FINAL\_16B\_design.md Profile B 决策的合理性：

| 决策 | 22_FINAL 选择 | 全市场分布支持度 | 一致性 |
|---|---|---|---|
| K=8 | ✓ | 24/66 = 36%（2025+ 主流） | **强支持** |
| 1 shared expert | ✓ | 24/66 = 36%（同上） | **强支持** |
| sigmoid + ALF | ✓ | 14/66 = 21%（但占 2025-H2 增量主体） | **方向正确** |
| MTP D=1 | △（仍在 wind tunnel） | 10/66 = 15% | **谨慎跟进** |
| GQA 16Q/4KV | ✓ | GQA 26/66 = 39% | **主流** |
| MLA | ✗ | 8/66 = 12%（DS 阵营专属） | **不引入是对的**（kernel 不开源） |
| Hybrid attention | ✗ | 9/66 = 14% | **不引入是对的**（infra 复杂度高） |
| HC / mHC | ✗ | 0/66（mHC 是 paper-only 内部实验） | **不引入是对的** |
| ~16B total / ~2.4B active / 1/6.5 稀疏 | ✓ | 与 V2-Lite / Moonlight / Ling-mini-1.0 / DeepSeekMoE-16B / mHC 3B 全部对齐 | **完美 anchor** |
| 第 1 层 dense | ✓ | V2-Lite / DeepSeekMoE / GLM-4.5-Air / ERNIE 4.5 21B 全用 | **共识** |
| `routed_scaling_factor=2.5` | ✓（在 spec） | V3 / dots1 / GLM-4.5 / GLM-4.6 一致 | **强支持** |

**结论**：22_FINAL Profile B 的每个决策都能在公开模型里找到 2+ 证据。**唯一仍待 wind tunnel 验证的是 MTP D=1**（2.4B active 是 boundary case，详见 23\_mtp\_investigation.md §8）。

---

## 6. wind tunnel A2 anchor 候选清单

> 1B/200M-active/25B tokens 量级（22\_FINAL §8 wind tunnel A2 规模）可参考的对照 anchors：

| 来源 | 总参 | Active | 训 tokens | 备注 |
|---|---|---|---|---|
| **mHC 3B 短训** | 2.97B | 612M | 39.3B | DeepSeek 内部最接近 A2 的锚点 |
| **DeepSeekMoE 2B** | 2.0B | 0.31B | 100B | DeepSeek 1.0 内部 anchor |
| OLMoE-1B-7B（小版） | – | – | – | 论文 §4 有更小 sweep |
| AttnRes 194M-528M anchors | 0.5-3.5B | 194-528M | 多档 | Kimi scaling law 锚点 |
| Ling-mini-1.0 small variant | ~3B | ~500M | ~50B | Inclusion AI 内部 sweep |
| **A2 自建** | **1B** | **200M** | **25B** | 目标对照点 |

→ **建议从 mHC 3B 短训配方起步**（LR=8.6e-4 / β=(0.9, 0.95) / ε=1e-20 / batch=320 / WSM），按 active 比例缩到 1B/200M。**注意 ε 选 OLMoE 派的 1e-8 还是 DeepSeek 派的 1e-20 要单独做 wind tunnel 决定 —— 这是一个未在主 spec §6 列出的开放变量**。

---

## 7. 仍需补足 / 数据存疑的项

| Model | 缺什么 | 优先级 |
|---|---|---|
| Hunyuan-Turbo-S | 完整 expert / hidden 配置（论文未明确） | M |
| Hy3 (Hunyuan-3.0) | 全部数据，仅 preview | M |
| Llama 4 Behemoth | 尚未发布 | L |
| Intern-S1-Pro | 详细 config | M |
| Seed1.5-VL / Seed-Thinking-v1.5 | 论文公开但 config 内嵌 | L |
| Switch Transformer XXL | head\_dim 等 attention 细节 | L（历史模型） |
| GLaM | 完整 head 数 / FFN | L |
| Pangu Pro MoGE | grouped routing 公式细节 | M |
| Step-3 | MFA 公式 + group 配置 | M |
| MiniMax-M2 D=3 MTP 配置 | λ schedule 细节 | M |

---

## 8. 与本仓库其他笔记的交叉引用

- **架构选型论证**：22\_FINAL\_16B\_design.md（主 spec）+ 23\_mtp\_investigation.md（MTP 决策）+ 25\_node\_limited\_routing.md（NLR 决策）
- **新机制评估**：26\_attention\_residuals.md（AttnRes）+ 27\_mhc.md（mHC）
- **scaling law 与稀疏度**：17\_finegrained\_scaling.md + 18\_params\_vs\_flops.md + 21\_reasoning\_vs\_memorization.md
- **训练动态**：19\_sparse\_upcycling.md + 20\_mtp\_gloeckle.md
- **个体论文深读 (16B-100B)**：01-16（DeepSeekMoE / V2 / V3 / ALF / Qwen3 / K2 / Hunyuan / Ling / OLMoE / Mixtral / Skywork / Yuan-M32 / MiniMax / Jamba / Step-3）+ 24\_dots1.md
- **个体论文深读 (100B-1T)**：35\_glm45.md（深>宽 355B）+ 36\_longcat.md（zero-experts 560B + ScMoE）+ 37\_ling1t.md（Ling Scaling Law + WSM + FP8 1T）+ 40\_qwen3_next.md（Hybrid DeltaNet 80B/3B）+ 41\_dsa.md（DeepSeek Sparse Attention V3.2）
- **优化器 / 训练系统**：39\_muon.md（Muon vs AdamW + Moonlight scaling law）
- **段位决策备忘**：38\_100b_to_200b_gap.md（为什么市场跳过 200B）+ 42\_100b_cookbook.md（100B+ 12 步决策树）
