#!/usr/bin/env python3
"""Build a single-file HTML viewer for all MoE paper notes + the 16B design doc."""
import base64
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPERS_DIR = ROOT / "papers"
OUT = ROOT / "index.html"

GROUPS = [
    ("📖 入门读本", ["31_foundations"]),
    ("DeepSeek 家族（基础）", ["01_deepseekmoe", "02_deepseek_v2", "03_auxloss_free", "04_deepseek_v3"]),
    ("中国头部 MoE", ["05_qwen3", "06_kimi_k2", "07_hunyuan_large", "08_ling_2", "24_dots1", "35_glm45", "36_longcat", "37_ling1t", "40_qwen3_next", "41_dsa", "44_step35_flash"]),
    ("开源 / 研究 MoE", ["09_olmoe", "10_mixtral", "11_skywork_moe", "12_yuan_m32"]),
    ("架构创新", ["13_minimax_01", "14_minimax_m1", "15_jamba", "16_step3", "26_attention_residuals", "27_mhc"]),
    ("Scaling Laws & 基础", ["17_finegrained_scaling", "18_params_vs_flops", "19_sparse_upcycling", "20_mtp_gloeckle", "21_reasoning_vs_memorization", "39_muon"]),
    ("⭐ 最终设计", ["22_FINAL_16B_design"]),
    ("📋 专项调研", ["23_mtp_investigation", "25_node_limited_routing", "28_open_source_moe_catalog", "29_wind_tunnel_a2", "30_routing_implementation", "32_depth_width_tradeoff", "33_advanced_concepts", "34_head_dim_deep_dive", "38_100b_to_200b_gap", "42_100b_cookbook", "43_short_wide_design"]),
]

def extract_title(text: str) -> str:
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return ""

def extract_subtitle(text: str) -> str:
    m = re.search(r"\*\*arXiv\*\*:\s*([0-9.]+)", text)
    return m.group(1) if m else ""

papers = []
for group, ids in GROUPS:
    for pid in ids:
        path = PAPERS_DIR / f"{pid}.md"
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        content = path.read_text(encoding="utf-8")
        papers.append({
            "id": pid,
            "group": group,
            "title": extract_title(content),
            "arxiv": extract_subtitle(content),
            "b64": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        })

sidebar_html_parts = []
for group, ids in GROUPS:
    sidebar_html_parts.append(f'<div class="group-header">{group}</div>')
    for pid in ids:
        p = next((x for x in papers if x["id"] == pid), None)
        if not p:
            continue
        sidebar_html_parts.append(
            f'<a href="#{p["id"]}" class="paper-link" data-id="{p["id"]}">'
            f'<span class="paper-num">{pid.split("_")[0]}</span>'
            f'<span class="paper-title-side">{p["title"]}</span>'
            + (f'<span class="paper-arxiv">{p["arxiv"]}</span>' if p["arxiv"] else "")
            + "</a>"
        )

sidebar_html = "\n".join(sidebar_html_parts)
js_payload = json.dumps({p["id"]: p["b64"] for p in papers}, ensure_ascii=False)

header_title = "MoE 论文阅读笔记 — 16B 总参 MoE 设计参考"
header_sub = f"{len(papers)-5} 篇一手论文 + 1 篇最终设计 + 4 篇专项决策备忘（含 100B+ Cookbook + 200B 真空带 + Muon vs AdamW + DSA + Hybrid Attention）"

# NOTE: this string is built without f-string interpolation for the script block,
# so we don't need to double-escape JS curly braces. We only interpolate via
# str.replace() for known keys.
TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__HEADER_TITLE__</title>

<!-- Marked (Markdown) -->
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<!-- highlight.js for code -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/styles/github.min.css" id="hljs-light-css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/styles/github-dark.min.css" id="hljs-dark-css" disabled>
<script src="https://cdn.jsdelivr.net/npm/highlight.js@11.10.0/lib/index.min.js"></script>

<!-- KaTeX for math (faster + more robust than MathJax for marked-katex-extension) -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js"></script>

<!-- GitHub-flavored CSS basics; we override below -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.5.1/github-markdown-light.css" id="gh-md-light">

<style>
:root {
  --sidebar-w: 320px;
  --header-h: 60px;
  --bg: #fafbfc;
  --sidebar-bg: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --accent-light: #ddf4ff;
  --text: #1f2328;
  --text-dim: #656d76;
  --code-bg: #f6f8fa;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; height: 100%;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: var(--text);
  background: var(--bg);
  font-size: 15px;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}
header {
  height: var(--header-h);
  background: white;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 24px;
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 10;
}
header .title { font-size: 16px; font-weight: 600; }
header .subtitle { font-size: 13px; color: var(--text-dim); margin-left: 12px; }
header .search-wrap { margin-left: auto; display: flex; align-items: center; gap: 12px; }
header input[type="search"] {
  width: 240px; padding: 6px 12px;
  border: 1px solid var(--border); border-radius: 6px;
  font-size: 13px; font-family: inherit; background: white; color: var(--text);
}
header button.theme-toggle {
  background: transparent; border: 1px solid var(--border);
  padding: 6px 12px; border-radius: 6px; cursor: pointer; font-size: 13px;
  color: var(--text);
}
.layout { display: flex; padding-top: var(--header-h); min-height: 100vh; }
.sidebar {
  width: var(--sidebar-w); background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  position: fixed; top: var(--header-h); left: 0; bottom: 0;
  overflow-y: auto; padding: 16px 0;
}
.group-header {
  padding: 12px 20px 4px;
  font-size: 11px; font-weight: 700; color: var(--text-dim);
  text-transform: uppercase; letter-spacing: 0.5px; margin-top: 8px;
}
.group-header:first-child { margin-top: 0; }
.paper-link {
  display: grid; grid-template-columns: 28px 1fr; gap: 0 8px;
  padding: 8px 20px; text-decoration: none; color: var(--text);
  font-size: 13px; line-height: 1.4;
  border-left: 3px solid transparent; transition: background 0.15s;
}
.paper-link:hover { background: var(--accent-light); }
.paper-link.active { background: var(--accent-light); border-left-color: var(--accent); font-weight: 500; }
.paper-num { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: var(--text-dim); padding-top: 2px; }
.paper-title-side { font-weight: 500; }
.paper-arxiv { grid-column: 2 / 3; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; color: var(--text-dim); margin-top: 1px; }
.content {
  margin-left: var(--sidebar-w);
  flex: 1;
  padding: 32px 48px 80px;
  max-width: calc(100vw - var(--sidebar-w));
  overflow-x: hidden;
}
.markdown-body {
  max-width: 980px;
  margin: 0 auto;
  background: transparent !important;
  color: var(--text) !important;
  font-family: inherit !important;
  font-size: 15px !important;
  line-height: 1.75 !important;
}
.markdown-body h1, .markdown-body h2, .markdown-body h3,
.markdown-body h4, .markdown-body h5, .markdown-body h6 {
  color: var(--text) !important;
  border-bottom-color: var(--border) !important;
  scroll-margin-top: calc(var(--header-h) + 12px);
}
.markdown-body h1 {
  border-bottom: 2px solid var(--border);
  padding-bottom: 12px;
  font-size: 28px;
  margin: 0 0 24px 0;
}
.markdown-body h2 { margin-top: 36px; font-size: 22px; padding-bottom: 6px; }
.markdown-body h3 { margin-top: 28px; font-size: 18px; }
.markdown-body h4 { margin-top: 20px; font-size: 16px; }
.markdown-body p { margin: 12px 0; }
.markdown-body ul, .markdown-body ol { padding-left: 28px; }
.markdown-body li { margin: 4px 0; }
.markdown-body blockquote {
  color: var(--text-dim) !important;
  border-left: 4px solid var(--border) !important;
  background: var(--code-bg);
  padding: 8px 16px;
  margin: 12px 0;
}

/* Tables: wrap in scrollable container so they don't break layout */
.table-wrap { overflow-x: auto; margin: 16px 0; }
.markdown-body table {
  display: table;
  width: auto;
  border-collapse: collapse;
  border-spacing: 0;
  font-size: 14px;
  background: transparent !important;
}
.markdown-body th, .markdown-body td {
  padding: 8px 12px;
  border: 1px solid var(--border);
}
.markdown-body th {
  background: var(--accent-light) !important;
  font-weight: 600;
  white-space: nowrap;
}
.markdown-body tr { background: transparent !important; }
.markdown-body tr:nth-child(2n) { background: var(--code-bg) !important; }

/* Code */
.markdown-body code {
  background: var(--code-bg) !important;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 13px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color: var(--text) !important;
}
.markdown-body pre {
  background: var(--code-bg) !important;
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 16px;
  overflow-x: auto;
  line-height: 1.5;
}
.markdown-body pre code {
  padding: 0;
  background: transparent !important;
  font-size: 13px;
}

/* KaTeX tweaks */
.katex { font-size: 1.0em; }
.katex-display { margin: 16px 0; overflow-x: auto; overflow-y: hidden; }

.welcome { text-align: center; padding: 100px 20px; color: var(--text-dim); }
.welcome h2 { color: var(--text); }

.toc-fab {
  position: fixed; right: 20px; bottom: 20px;
  background: var(--accent); color: white;
  width: 44px; height: 44px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 5; border: none; font-size: 18px;
}
.toc-panel {
  position: fixed; right: 20px; bottom: 76px;
  width: 320px; max-height: 60vh;
  background: white; border: 1px solid var(--border);
  border-radius: 8px; box-shadow: 0 8px 24px rgba(0,0,0,0.15);
  overflow-y: auto; display: none; padding: 12px 16px; z-index: 5;
}
.toc-panel.open { display: block; }
.toc-panel h4 { margin: 0 0 8px 0; font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
.toc-panel a { display: block; padding: 4px 0; padding-left: 8px; font-size: 13px; color: var(--text); text-decoration: none; border-left: 2px solid transparent; line-height: 1.4; }
.toc-panel a:hover { color: var(--accent); }
.toc-panel a.toc-h3 { padding-left: 22px; color: var(--text-dim); font-size: 12px; }
.toc-panel a.toc-h4 { padding-left: 36px; color: var(--text-dim); font-size: 11px; }

@media (max-width: 980px) {
  :root { --sidebar-w: 100%; }
  .sidebar { position: relative; height: auto; top: 0; }
  .content { margin-left: 0; padding: 24px; max-width: 100vw; }
  header input[type="search"] { width: 140px; }
}

/* Dark theme */
body.dark {
  --bg: #0d1117;
  --sidebar-bg: #161b22;
  --border: #30363d;
  --accent: #58a6ff;
  --accent-light: #1f2d3d;
  --text: #e6edf3;
  --text-dim: #8b949e;
  --code-bg: #161b22;
}
body.dark header { background: #0d1117; }
body.dark header input[type="search"] { background: #0d1117; color: var(--text); }
body.dark .markdown-body th { background: var(--accent-light) !important; }
body.dark .markdown-body tr:nth-child(2n) { background: #161b22 !important; }
body.dark .markdown-body code { background: #161b22 !important; color: #f0f6fc !important; }
body.dark .markdown-body pre { background: #161b22 !important; }
body.dark .toc-panel { background: #161b22; }
body.dark .katex { color: var(--text); }

/* Print */
@media print {
  header, .sidebar, .toc-fab, .toc-panel { display: none !important; }
  .content { margin-left: 0; padding: 0; max-width: 100vw; }
  .markdown-body { max-width: 100%; }
}
</style>
</head>
<body>
<header>
  <div class="title">__HEADER_TITLE__</div>
  <div class="subtitle">__HEADER_SUB__</div>
  <div class="search-wrap">
    <a href="calc.html" style="color:var(--accent);text-decoration:none;font-size:13px;font-weight:500;margin-right:8px;">🧮 Calculator</a>
    <input type="search" id="search" placeholder="按标题搜索…" />
    <button class="theme-toggle" id="theme-toggle" title="切换主题">🌓</button>
  </div>
</header>
<div class="layout">
  <nav class="sidebar" id="sidebar">
__SIDEBAR_HTML__
  </nav>
  <main class="content">
    <article class="markdown-body" id="article">
      <div class="welcome">
        <h2>欢迎</h2>
        <p>左侧选择一篇论文开始阅读。建议从 <a href="#22_FINAL_16B_design">⭐ 22 - 16B MoE 最终设计建议</a> 开始。</p>
      </div>
    </article>
  </main>
</div>

<button class="toc-fab" id="toc-fab" title="本页目录">📑</button>
<aside class="toc-panel" id="toc-panel">
  <h4>本页目录</h4>
  <div id="toc-list"></div>
</aside>

<script id="papers-data" type="application/json">__PAYLOAD__</script>
<script>
const PAPERS = JSON.parse(document.getElementById('papers-data').textContent);

// Proper UTF-8 decoder for base64 (replaces deprecated escape()/decodeURIComponent hack)
function decodeUtf8B64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new TextDecoder('utf-8').decode(bytes);
}

// ---------- Math protection ----------
// Marked is told to ignore $...$ / $$...$$ by stashing math regions before parsing
// and restoring them afterwards, so $N_{routed}$ etc. are not mangled as italics.
function preprocessMath(md) {
  const stash = [];
  // Display $$...$$ first
  md = md.replace(/\$\$([\s\S]+?)\$\$/g, (m) => {
    stash.push(m);
    return '@@MATHBLOCK' + (stash.length - 1) + '@@';
  });
  // Inline $...$ (must not contain $ or newline)
  md = md.replace(/(?<![\$\\])\$([^\$\n]+?)\$(?!\d)/g, (m) => {
    stash.push(m);
    return '@@MATHINLINE' + (stash.length - 1) + '@@';
  });
  return { md, stash };
}
function restoreMath(html, stash) {
  return html.replace(/@@MATH(?:BLOCK|INLINE)(\d+)@@/g, (_, i) => stash[+i]);
}

// Configure marked
marked.setOptions({
  gfm: true,
  breaks: false,
  pedantic: false,
});
// Disable underscores being treated as emphasis inside identifiers (e.g. N_routed)
// Use the underscore-as-bold-only convention via a custom tokenizer extension.

// Wrap tables in scrollable div after render
function wrapTables(root) {
  root.querySelectorAll('table').forEach(tbl => {
    if (tbl.parentElement && tbl.parentElement.classList.contains('table-wrap')) return;
    const w = document.createElement('div');
    w.className = 'table-wrap';
    tbl.parentNode.insertBefore(w, tbl);
    w.appendChild(tbl);
  });
}

function loadPaper(id) {
  const b64 = PAPERS[id];
  if (!b64) return;
  let md = decodeUtf8B64(b64);
  const { md: clean, stash } = preprocessMath(md);
  let html = marked.parse(clean);
  html = restoreMath(html, stash);

  const article = document.getElementById('article');
  article.innerHTML = html;

  wrapTables(article);

  // Render code highlighting
  if (window.hljs) {
    article.querySelectorAll('pre code').forEach(b => {
      try { hljs.highlightElement(b); } catch (e) {}
    });
  }

  // Render math (KaTeX auto-render)
  if (window.renderMathInElement) {
    try {
      renderMathInElement(article, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
        errorColor: '#cc0000',
        strict: false,
      });
    } catch (e) { console.warn('KaTeX render error', e); }
  }

  // mark active
  document.querySelectorAll('.paper-link').forEach(a => a.classList.remove('active'));
  const link = document.querySelector(`.paper-link[data-id="${id}"]`);
  if (link) {
    link.classList.add('active');
    link.scrollIntoView({ block: 'nearest' });
  }

  buildTOC(article);
  window.scrollTo(0, 0);
}

function slug(s) {
  return s.toLowerCase()
    .replace(/[^\w一-鿿]+/g, '-')
    .replace(/^-|-$/g, '');
}

function buildTOC(article) {
  const list = document.getElementById('toc-list');
  list.innerHTML = '';
  const headers = article.querySelectorAll('h2, h3, h4');
  headers.forEach((h, idx) => {
    if (!h.id) h.id = 'h-' + idx + '-' + slug(h.textContent).slice(0, 40);
    const a = document.createElement('a');
    a.href = '#' + h.id;
    a.textContent = h.textContent;
    a.className = h.tagName === 'H3' ? 'toc-h3' : (h.tagName === 'H4' ? 'toc-h4' : '');
    a.onclick = (e) => {
      e.preventDefault();
      document.getElementById(h.id).scrollIntoView({ behavior: 'smooth', block: 'start' });
    };
    list.appendChild(a);
  });
}

// Sidebar links
document.querySelectorAll('.paper-link').forEach(a => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    const id = a.getAttribute('data-id');
    history.pushState(null, '', '#' + id);
    loadPaper(id);
  });
});

function onHashChange() {
  const hash = location.hash.replace('#', '');
  if (hash && PAPERS[hash]) loadPaper(hash);
}
window.addEventListener('hashchange', onHashChange);

document.getElementById('search').addEventListener('input', (e) => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('.paper-link').forEach(a => {
    const t = a.querySelector('.paper-title-side').textContent.toLowerCase();
    a.style.display = (!q || t.includes(q)) ? 'grid' : 'none';
  });
});

document.getElementById('toc-fab').addEventListener('click', () => {
  document.getElementById('toc-panel').classList.toggle('open');
});
document.addEventListener('click', (e) => {
  if (!e.target.closest('.toc-panel') && !e.target.closest('.toc-fab')) {
    document.getElementById('toc-panel').classList.remove('open');
  }
});

// Theme toggle
const themeToggle = document.getElementById('theme-toggle');
const stored = localStorage.getItem('moe-theme');
function applyTheme(dark) {
  document.body.classList.toggle('dark', dark);
  document.getElementById('hljs-light-css').disabled = dark;
  document.getElementById('hljs-dark-css').disabled = !dark;
}
applyTheme(stored === 'dark');
themeToggle.addEventListener('click', () => {
  const dark = !document.body.classList.contains('dark');
  applyTheme(dark);
  localStorage.setItem('moe-theme', dark ? 'dark' : 'light');
});

// Initial load
if (location.hash && PAPERS[location.hash.replace('#', '')]) {
  onHashChange();
} else {
  setTimeout(() => loadPaper('22_FINAL_16B_design'), 50);
}
</script>
</body>
</html>
"""

html = (TEMPLATE
        .replace("__HEADER_TITLE__", header_title)
        .replace("__HEADER_SUB__", header_sub)
        .replace("__SIDEBAR_HTML__", sidebar_html)
        .replace("__PAYLOAD__", js_payload))

OUT.write_text(html, encoding="utf-8")
print(f"OK -> {OUT}  ({OUT.stat().st_size:,} bytes)")
print(f"Papers indexed: {len(papers)}")
