# MoE 论文阅读笔记 — 16B 总参 MoE 设计参考

44 篇一手论文 + 1 篇 16B 最终设计 + 5 篇专项决策备忘的 MoE 架构调研合集，覆盖 16B-1T 全段位。

## 在线阅读

GitHub Pages 自动部署：访问 `https://pixeli99.github.io/moe/`

## 本地构建

```bash
python3 build_html.py
# 输出 index.html，浏览器直接打开
```

## 笔记结构

```
papers/
  01-21          # 基础论文（DeepSeek 家族 / Mixtral / Jamba / scaling law 等）
  22_FINAL_*     # 16B 最终设计 spec
  23-34          # 专项调研（MTP / NLR / wind tunnel / depth-width 等）
  35-37, 40-41   # 100B-1T 段位个体论文（GLM-4.5 / LongCat / Ling-1T / Qwen3-Next / V3.2 DSA）
  38             # 100B-200B 真空带分析
  39             # Muon 优化器深度
  42             # 100B+ MoE 设计 Cookbook（12 步决策树）
  43             # 16B 矮胖 MoE 设计（教学版）
  44             # Step 3.5 Flash（200B/45L SWA Hybrid）
```

## 推荐阅读顺序

1. **入门** → `31_foundations` + `28_open_source_moe_catalog`
2. **架构选型** → `42_100b_cookbook` (12 步决策树)
3. **scaling law** → `17_finegrained_scaling` + `18_params_vs_flops` + `37_ling1t`
4. **16B 设计** → `22_FINAL_16B_design` + `43_short_wide_design`
5. **100B+ 段位** → `35_glm45` / `36_longcat` / `37_ling1t` / `44_step35_flash`
6. **稳定性 / 优化器** → `39_muon` + `36_longcat` (hidden z-loss)

## 部署

每次 push 到 `main` 分支会触发 `.github/workflows/deploy.yml`，自动：
1. 运行 `build_html.py` 重新生成 `index.html`
2. 上传到 GitHub Pages

如需手动触发，去仓库 Actions 标签页点 "Run workflow"。
