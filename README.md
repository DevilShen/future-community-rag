# 未来社区 · 智慧物业知识库问答系统

一个**私有知识库问答系统**：把物业 / 未来社区领域的五类文档入库，入库前做「筛选存储」（脱敏、去重、分类打标），再通过 RAG 支持自然语言问答，回答带来源引用。前端是移动端优先的**侧栏工作台**：桌面端左侧展示知识库统计、分类筛选与历史会话，右侧问答；窄屏时侧栏收成抽屉，手机可直接访问。

## 核心亮点

- **筛选存储，而非「入库即存」**：入库前对整批语料做一次脱敏（身份证/手机号/邮箱/敏感词/金额）+ 跨文档去重 + 分类打标，这是本项目区别于普通 RAG demo 的关键。
- **全链路可离线跑通**：`EMBED_PROVIDER=mock` + `LLM_PROVIDER=mock` 时，向量化与作答都不调任何外部接口，一条命令即可本地演示整条链路。
- **供应商可切换**：统一用 `openai` SDK，靠 `base_url` 区分 DeepSeek（对话）与 SiliconFlow（向量），换模型只改 `.env`。
- **流式问答**：`/api/ask` 走 SSE，先推来源、再逐字推正文，前端有打字机效果。
- **答案可溯源**：每个结论标 `[1][2]` 引用，前端展示来源文件、分类与匹配度；点开来源卡片还能看到**原文位置**——PDF 第几页、docx 的章节路径与段落序号，外加命中片段全文，省去在几十页原文里翻找。
- **引用可点开**：来源卡片上的「原文 ›」一键打开该文档的阅读视图（桌面浮层 / 手机全屏），自动滚到命中片段并高亮，底部常驻「回到命中处」；PDF 按「页」分节、docx 按「章节」分节，去重删掉的片段也会显式标注缺口。
- **工作台式前端**：侧栏一屏展示知识库规模与五类分布、分类筛选、历史会话（localStorage 持久化，刷新不丢）；答案按 Markdown 排版（列表 / 表格 / 代码块），窄屏自动收成抽屉。零前端框架、零构建步骤。

## 架构

系统分两条链路：**入库链路**离线跑一次，**问答链路**每次提问在线跑。

```
【入库链路】离线 · 一次性
─────────────────────────────────────────────────────────────────────────
 corpus/                    rag/loader.py    rag/chunker.py   rag/screener.py
 ├── 01_行业标准/            解析              中文分块          筛选
 ├── 02_交付流程/    ──▶     docx / pdf   ──▶  500 字一块  ──▶  ① 脱敏
 ├── 03_操作手册/            md / txt          80 字重叠         ② 去重
 ├── 04_交付文档/            （扫描版 PDF                      ③ 分类打标
 └── 05_过往案例/             会提示正文为空）                     ▲
                                                          整批一起做，
 corpus_samples/  ── 虚构示例，不自动入库 ──▶             才能发现跨文档重复
                                                               │
                                                    rag/embedder.py
                                                    bge-m3（1024 维）/ mock
                                                               │
                                                    rag/store.py → ChromaDB
                                                               ▲
                              ingest.py 串起全流程 ─────────────┘
                              python ingest.py [--rebuild] [--category] [--mask-amounts] [--dry-run]

【问答链路】在线 · 每次提问
─────────────────────────────────────────────────────────────────────────
 static/ H5         POST /api/ask                api.py (FastAPI)
 index.html   ──▶   {"question",     ──▶        /api/ask  /api/categories
 style.css          "category",                 /api/health  /api/doc
 app.js             "top_k"}                    /  → index.html
                                                      │
                                                      ▼
                                        rag/retriever.py
                                        向量化问题 → ChromaDB top-k → 阈值过滤
                                                      │
                                                      ▼
                                        rag/llm.py
                                        format_context 编号 → 拼提示词 → 模型作答
                                                      │
                                                      ▼
                                        SSE 事件序列（流式）
                                        sources → delta×N → done
                                                      │
                                                      ▼
                                        H5 渲染：正文 + [1] 引用 + 匹配度
```

**为什么筛选要整批做**：重复的两块天然分处不同文件，逐个文件筛选只能看到文件内部，
跨文档重复永远发现不了。所以管道分三段——逐文件解析分块 → 全语料一次性筛选 →
按文件回写，既拿到全局去重，又保留逐文件进度与失败隔离。

## 目录结构

```
future-community-rag/
├── config.py            # 全局配置（.env → 单例对象）
├── ingest.py            # 入库管道入口
├── api.py               # FastAPI 问答后端
├── requirements.txt
├── .env.example
├── .gitignore
├── rag/
│   ├── loader.py        # 文档解析（docx/pdf/md/txt）+ 位置锚点（页码 / 章节 / 段落）
│   ├── chunker.py       # 中文分块（带重叠）+ 块位置反查
│   ├── screener.py      # 筛选：脱敏 / 去重 / 打标
│   ├── embedder.py      # 向量化（SiliconFlow / mock）
│   ├── store.py         # ChromaDB 读写
│   ├── retriever.py     # 检索 + 上下文编排
│   └── llm.py           # 对话作答（DeepSeek / mock）
├── corpus/              # 真实语料目录（五个分类子目录，公开版不含）
│   ├── 01_行业标准/ 02_交付流程/ 03_操作手册/ 04_交付文档/ 05_过往案例/
├── corpus_samples/      # 虚构示例语料，不自动入库（见该目录 README）
├── static/              # H5 前端（index.html / style.css / app.js）
└── chroma_db/           # 向量库持久化目录（入库后生成）
```

> `corpus_samples/` 是**虚构示例**，刻意放在 `corpus/` 之外，`ingest.py` 不会读到它。
> 空库演示时可手动拷入，详见 [`corpus_samples/README.md`](corpus_samples/README.md)。

### 公开版本的匿名化说明

本仓库是公开版本，做了两处处理，**不影响代码运行**：

- **真实语料不入库**：`corpus/`（行业标准 / 交付流程 / 操作手册 / 交付文档 / 过往案例）
  为内部文档，已在 `.gitignore` 中整体排除，仅保留在作者本地。仓库内只含
  `corpus_samples/` 的虚构示例，所以克隆后直接 `python ingest.py` 入库为空，属预期行为。
- **评测集已匿名化**：`eval/questions.jsonl` 中的客户名、项目名、承建/集成单位、部署端口
  与文档文件名均已替换为中性占位（如「示例社区」「示例e家」「示例_安装部署手册.docx」）。
  因此其中的 `gold_sources` 指向的是**匿名化前的内部语料路径**，与仓库内文件并不对应——
  该文件保留的是评测集的**结构与考点**，不是可开箱复现的数据集。

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 | Python 3.11 |
| 对话模型 | `deepseek-v4-flash`（公司中转站，OpenAI 兼容；见下「接入公司中转站」） |
| 向量模型 | SiliconFlow `BAAI/bge-m3`（1024 维） |
| 向量库 | ChromaDB（本地持久化，余弦距离） |
| 后端 | FastAPI + Uvicorn |
| 前端 | 原生 H5（HTML + CSS + JS），移动端优先，无框架 |

## 安装

```bash
# 1. 创建并激活虚拟环境（Python 3.11）
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥
cp .env.example .env
# 编辑 .env，填入真实的 LLM_API_KEY / EMBED_API_KEY
```

## 配置（.env 关键项）

| 变量 | 说明 | 缺省 |
|---|---|---|
| `LLM_PROVIDER` | 对话模型：`deepseek` / `mock` | `deepseek` |
| `LLM_API_KEY` | 对话模型密钥 | — |
| `LLM_BASE_URL` | 兼容端点（中转站需带 `/v1`） | `https://api.deepseek.com` |
| `LLM_MODEL` | 对话模型名 | `deepseek-chat` |
| `LLM_MAX_TOKENS` | 单次生成上限；推理型模型需 ≥2048 | `2048` |
| `LLM_USER_AGENT` | 自定义 UA，留空即不附加；仅中转站白名单场景需要 | 空 |
| `EMBED_PROVIDER` | 向量：`siliconflow` / `mock` | `mock` |
| `EMBED_API_KEY` | 硅基流动密钥 | — |
| `EMBED_BASE_URL` | 硅基流动端点（需带 `/v1`） | `https://api.siliconflow.cn/v1` |
| `EMBED_MODEL` | 向量模型 | `BAAI/bge-m3` |
| `CHROMA_COLLECTION` | 向量集合名 | `future_community_kb` |
| `TOP_K` / `SCORE_THRESHOLD` | 检索条数 / 相似度下限 | `5` / `0.3` |
| `SENSITIVE_WORDS` | 自定义敏感词（逗号分隔） | 空 |

> 更多分块、脱敏、去重参数见 `.env.example` 内注释。运行 `python config.py` 可打印当前生效配置。

## 接入公司中转站

本项目当前指向公司中转站 `https://ai.ouflow.cn/v1`，模型 `deepseek-v4-flash`。接线与官方
DeepSeek 有三处差异，踩过才知道，记录在此：

**1. `base_url` 必须带 `/v1`。** 不带会 404 —— SDK 只会追加 `/chat/completions`，
不会替你补 `/v1`。

**2. 中转站没有 `deepseek-chat` 这个模型名。** 实测可用的 DeepSeek 系列是
`deepseek-v4-flash`（推理型）与 `deepseek-v4-pro`。

**3. `deepseek` 系列被客户端白名单拦住，需伪装 UA。** 只要 `LLM_USER_AGENT` 为空，
请求一律被拒：

```
403 {"code":"agent_not_allowed",
     "message":"模型 deepseek-flash 仅限在 zcode / claude-code / codex / ... 中使用"}
```

实测**只有** `User-Agent: claude-cli/1.0.0` 能过；换 `claude-code/2.0.0`、`codex-cli`、
`zcode` 或走 Anthropic 协议 `/v1/messages` 都被拒。因此 `.env` 里配了：

```ini
LLM_USER_AGENT=claude-cli/1.0.0
```

> ⚠️ **合规提示。** 这个头让请求看起来像白名单里的客户端，**构成绕过公司的访问控制**。
> 启用前请确认这么绕符合公司规定；若不合规，可改用同一中转站上**不需要伪装**的模型
> （实测 `k3` / `kimi-for-coding` 直接可用），只需把 `LLM_MODEL` 换成它、并把
> `LLM_USER_AGENT` 留空即可，无需改代码。

**推理型模型的坑。** `deepseek-v4-flash` 会先在 `reasoning_content` 里思考，这部分**同样
计入 `max_tokens`**。预算给小了会出现 `finish_reason=length` 且**正文一个字都没有**。
本项目已在 `rag/llm.py` 中显式识别该情况并抛出可读错误（而非静默返回空串）。
实测一次 6 点归纳回答消耗 2459 tokens（思考 + 正文），故 `LLM_MAX_TOKENS` 取 **4096**。
上文检索命中的片段越长，思考越费 token，若偶发截断，继续调大该值。

**没配密钥也不会报错。** 若 `LLM_API_KEY` 为空，问答不中断，而是降级为
**仅检索模式**：直接把检索到的原文片段连同 `[1][2]` 编号返回，前端徽标显示「仅检索模式」。
这样「入库 + 检索」这条链路在没有大模型的情况下依然可演示、可验收。

## 入库

把 `docx / pdf / md / txt` 文档放进 `corpus/` 对应的五个分类子目录，然后：

```bash
python ingest.py                 # 增量入库全部语料
python ingest.py --rebuild       # 清空向量库后全量重建
python ingest.py --category past_case   # 只入库「过往案例」分类
python ingest.py --mask-amounts  # 额外脱敏金额（合同/报价类语料建议加）
python ingest.py --dry-run       # 只解析+筛选，不写库，用于检查筛选效果
```

入库管道五步：**解析 → 中文分块 → 筛选（脱敏/去重/打标）→ 向量化 → 写入 ChromaDB**。筛选按整批语料做（而非逐文件），确保能发现跨文档重复。

**引用定位是怎么来的。** 解析阶段不再把整篇文档拍成一个大字符串，而是逐片保留位置：
PDF 一片一页、docx 一片一段（按**文档流顺序**遍历，表格回到它真正所属的章节里）。分块时
用块在正文中的字符偏移二分反查，得出「第几页 / 章节路径 / 章节内第几段」，写进 ChromaDB 元数据。
前提是逐片规整后的拼接与整串规整**逐字符一致**——`rag/loader.py:_build_anchored` 对此有断言，
一旦不一致就打印告警并**降级为无位置信息**（正文照常可用，不影响问答）。

> ⚠️ **位置信息依赖元数据 schema，改动后必须重新入库。** 本次变更前入库的旧块没有
> `page / heading_path / para_index` 三个键，前端会显示「该格式无位置信息」。补齐全库：
> `python ingest.py`（增量即可，写前会按来源先删旧块），**然后重启 `api.py`**。
> `md / txt` 本身没有页与章节概念，永远显示「该格式无位置信息」，这是预期行为。

> ⚠️ **入库后必须重启 `api.py`，否则检索不到新数据。**
> `api.py` 的 `VectorStore` 是进程级单例，ChromaDB 的 HNSW 索引在启动时载入内存；
> `ingest.py` 是另一个进程，往磁盘写入的新块不会反映到已运行 API 的内存索引里。
> 表现为「刚入库的文档提问仍答未收录」。**入库和重启是两个独立步骤，缺一不可。**

关于语料目录：

- **支持的后缀**：`.docx / .pdf / .md / .txt`。其余格式（`.doc`、`.pptx`、`.xls(x)`、`.zip` 等）
  会被跳过，并打印数量与后缀汇总 —— 不会静默忽略。`.doc` 是 Word 97-2003 二进制格式，
  python-docx 读不了，需先另存为 `.docx`，或用 LibreOffice 无头转换。
- **分类按「相对语料根的第一层目录」判定**，支持多层子目录：
  `corpus/04_交付文档/某项目/商务文档/x.docx` → `delivery_doc`。
  多深都行，但第一层目录名必须正好是五个分类目录之一。
- **扫描版 PDF**（无文字层）会因抽不出文字而失败并报出，需 OCR，本项目不处理。

## 启动

```bash
# 方式一：直接运行
python api.py

# 方式二：开发期热重载
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

浏览器打开 `http://127.0.0.1:8000`；手机与电脑同局域网时，访问 `http://<电脑IP>:8000` 即可。

> ⚠️ **Windows 上建议用 `127.0.0.1` 而不是 `localhost`。** 服务监听的是 `0.0.0.0`
> （仅 IPv4），而 Windows 的 hosts 把 `localhost` 同时指向 `127.0.0.1` 和 `::1`；
> 浏览器若先试 IPv6 的 `::1`，那里没有监听，就会连不上，表现为前端徽标显示
> **「无法连接后端」**。实测本机 `::1:8000` 拒绝连接、`127.0.0.1:8000` 正常。
> 前端已带 5 次重试，全部失败时把真实错误写进徽标的 `title`（鼠标悬停可见），
> 并按 `F12` 在 Console 里能看到具体失败的 URL。

## 离线 mock 跑通（无需任何密钥）

在 `.env` 中设置：

```dotenv
EMBED_PROVIDER=mock
LLM_PROVIDER=mock
```

此时向量用哈希生成、答案用检索片段拼模板，**全程不联网**。先 `python ingest.py` 入库，再 `python api.py` 启动即可完整演示。注意：mock 向量模式使用更低的相似度阈值 `SCORE_THRESHOLD_MOCK`（`config.effective_score_threshold()` 会自动切换），否则离线检索会为空。

## API 接口

### `GET /api/health`
服务状态、mock 开关、库内块数、模型名。

### `GET /api/categories`
```json
{
  "total": 128,
  "categories": [
    {"id": "industry_standard", "name": "行业标准", "count": 30},
    {"id": "delivery_process", "name": "交付流程", "count": 22}
  ]
}
```

### `GET /api/doc`
按来源取该文档**入库后的全部片段**，供前端「查看原文」浮层渲染阅读视图。

```
GET /api/doc?source=03_操作手册/示例_社区管理后台操作手册.docx
```

```json
{
  "source": "03_操作手册/…docx",
  "title": "示例社区数字化建设项目",
  "category": "operation_manual",
  "category_name": "操作手册",
  "ext": ".docx",
  "mask_hits": 0,
  "chunk_count": 108,
  "chunks": [
    {"chunk_id": "…", "chunk_index": 0, "text": "…",
     "page": 0, "heading_path": "登陆注册 > 忘记密码", "para_index": 1, "mask_hits": 0}
  ]
}
```

- 数据**只来自向量库**，不读原始文件：既没有 `source` 参数带来的路径穿越面，也保证
  「前端看到的 = 模型检索到的 = 已脱敏后的内容」；
- `ext` 由 source 后缀推出，前端据此决定按「页」还是按「章节」分节；
- 该文档一块都查不到时返回 **404**（未入库 / 整篇被去重），而不是返回空文档——
  空文档会让浮层白屏，分不清是「没入库」还是「接口坏了」；
- 大文档响应较大（最大一篇 483 块 / 未压缩约 780KB），服务端已挂 `GZipMiddleware`
  压缩到约 **23%**（181KB），实测手机同网可接受。

### `POST /api/ask`（SSE 流式）
请求体：
```json
{"question": "承接查验包括哪些内容？", "category": "delivery_process", "top_k": 5}
```

响应为 `text/event-stream`，事件序列固定：

1. `sources`：来源列表，每项为
   `{index, title, source, chunk_id, chunk_index, category, score, snippet, page, heading_path, para_index, text}`
   —— `snippet` 是折叠态的 140 字摘要，`text` 是展开态用的命中全文（上限 800 字）；
   `page / heading_path / para_index` 是位置三件套，三者皆空时前端显示「该格式无位置信息」；
   `chunk_id / chunk_index` 是「查看原文」用来精确高亮那一个块的锚点
2. `delta`：正文增量（可多次）
3. `done`：`{model, provider}` 收尾
4. `error`（异常时）：`{message}`

命令行验证：
```bash
curl -N -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"承接查验包括哪些内容？"}'
```

## 手机端 H5 部署路径

前端与后端**同源**（`api.py` 同时托管 `static/`），所以不存在跨域配置问题。
按环境分三步走：

### 第 1 步 · 本地自测（电脑浏览器）

```bash
python api.py          # 或 uvicorn api:app --reload
```

打开 `http://localhost:8000`。前端所有请求都是相对路径（`/api/ask`、`/static/app.js`），
无需任何额外配置。

### 第 2 步 · 局域网手机访问（演示常用）

```bash
python api.py          # .env 里 API_HOST=0.0.0.0 即监听所有网卡（默认已是）
```

1. 查电脑 IP：Windows 用 `ipconfig`，macOS/Linux 用 `ifconfig` 或 `ip addr`；
2. 手机连**同一个 Wi-Fi**，浏览器访问 `http://<电脑IP>:8000`；
3. 首次可能被系统防火墙拦截，放行 8000 端口即可。

> 提示：H5 已按移动端优先设计（`viewport` 已配、窄屏下侧栏自动收成抽屉，点左上角 ☰ 展开），
> 直接用手机浏览器打开即可，不需要装 App；也可「添加到主屏幕」当轻应用用。
> 桌面端（≥900px）为双栏工作台，适合投影演示。

### 第 3 步 · 部署到服务器（加域名即可对外）

因为同源，**只需把域名指向这台服务器**，不用改任何前端代码。

```bash
# 服务器上（建议用 systemd 或 supervisor 守护）
uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

```nginx
# /etc/nginx/conf.d/kb.conf
server {
    listen 80;
    server_name kb.example.com;          # ← 换成你的域名

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # 关键：SSE 必须关闭缓冲，否则流式问答会「攒一大堆再一次性吐出」
        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 300s;         # 推理型模型首字延迟较长，超时给足
        chunked_transfer_encoding on;
    }
}
```

再配 HTTPS（`certbot --nginx` 一条命令即可）。配好 HTTPS 后浏览器就允许
「添加到主屏幕」，体验与 App 接近。

**部署后必须核对的三项**：

| 项 | 要求 | 怎么验 |
|---|---|---|
| `API_HOST` | 生产建议 `127.0.0.1`（只让 Nginx 访问，不直接暴露） | `python config.py` 看生效值 |
| `.env` / `chroma_db/` | 均已在 `.gitignore` 中，**绝不提交** | `git status` 应看不到它们 |
| SSE 未被缓冲 | `proxy_buffering off` + 接口自带的 `X-Accel-Buffering: no` | 提问时正文应逐字出现，而非等待后一次性出现 |

> 已知边界：演示语料规模（千级块）下 `category_counts()` 走全量元数据聚合；
> 若上到十万级，需在入库时维护计数表。

## 验收清单

分两大段：**先 mock 离线**（不依赖任何密钥和网络，证明代码本身是对的），
**再切真实 key**（证明外部接口对接是对的）。这样一旦出问题，
能立刻分清是「代码 bug」还是「密钥/网络/供应商问题」。

> 下面的实测值基于当前语料（`corpus/` 下 8 篇 PDF）。换成你自己的语料后
> 块数会变，**比例和现象**才是判断依据。

### 阶段 A · mock 离线跑通（无需任何密钥）

| # | 命令 | 预期结果 | 勾选 |
|---|---|---|---|
| A1 | `python -m venv .venv && .venv\Scripts\activate` | 虚拟环境激活 | ☐ |
| A2 | `pip install -r requirements.txt` | 无报错 | ☐ |
| A3 | `cp .env.example .env` 后设 `EMBED_PROVIDER=mock`、`LLM_PROVIDER=mock` | — | ☐ |
| A4 | `python config.py` | 「mock 向量模式」显示 `True` | ☐ |
| A5 | `python ingest.py --dry-run` | 8 篇成功 / 0 篇失败；打印「保留 N 块」 | ☐ |
| A6 | `python ingest.py` | 打印「入库 N 块」+ 库内分类分布条形图 | ☐ |
| A7 | `python api.py` | 输出 `Uvicorn running on http://0.0.0.0:8000` | ☐ |
| A8 | 浏览器开 `http://localhost:8000` | 页面加载，徽标显示「离线 mock 模式」 | ☐ |
| A9 | 输入「承接查验包括哪些内容？」 | 流式返回答案 + 来源卡片 + 匹配度 | ☐ |

**A 段通过标准**：全程不联网、不报错，问答有回答且带来源引用。
注意 mock 模式的匹配度天然偏低（哈希词袋），这是正常的，不是 bug。

### 阶段 B · 切真实 key 验证

| # | 操作 | 预期结果 | 勾选 |
|---|---|---|---|
| B1 | `.env` 设 `EMBED_PROVIDER=siliconflow` + `EMBED_API_KEY` | — | ☐ |
| B2 | `.env` 设 `LLM_PROVIDER=deepseek` + `LLM_API_KEY`（中转站另需 `LLM_BASE_URL` 带 `/v1` 与 `LLM_USER_AGENT`，见上文） | — | ☐ |
| B3 | `python config.py` | 「mock 向量模式」变为 `False`，密钥显示为 `agw_****xx` 打码形式 | ☐ |
| B4 | `python ingest.py --rebuild` | 重新入库，块数与 A6 一致 | ☐ |
| B5 | `python -m rag.llm "未来社区的验收流程分哪几个步骤？"` | 返回**模型归纳**的答案（非原文罗列），来源相似度 > 0.5 | ☐ |
| B6 | 浏览器提问同一问题 | 答案比 mock 模式**明显更像人话**，引用编号 `[1]-[5]` 与来源卡片对应 | ☐ |
| B7 | 提问一个**语料里没有**的问题（如「员工报销流程是什么」） | 回答「知识库中未收录相关内容」，**不编造** | ☐ |
| B8 | 点答案中的 `[1]` | 页面滚动定位到对应来源卡片 | ☐ |
| B9 | 点分类 chip「交付流程」后再提问 | 来源**只出现交付流程类文档**，不混入行业标准 | ☐ |

**B 段通过标准**：B5 是关键分界——若返回的是原文罗列而非归纳，说明其实还在 mock 模式，
检查 `LLM_PROVIDER`；B7 是防止模型编造的红线项，必须通过。

### 阶段 C · 筛选存储验证（本项目的核心亮点）

用 `corpus_samples/` 里的虚构示例验证脱敏，**不影响真实语料**：

| # | 命令 | 预期结果 | 勾选 |
|---|---|---|---|
| C1 | 把 `corpus_samples/05_过往案例/*.md` 拷进 `corpus/05_过往案例/` | — | ☐ |
| C2 | `python ingest.py --dry-run` | 「脱敏命中」**不为「无」**，应含 `身份证号×1、手机号×1、固定电话×1、电子邮箱×1` | ☐ |
| C3 | `python ingest.py --dry-run --mask-amounts` | 命中列表**额外出现**「金额×N」 | ☐ |
| C4 | `python ingest.py` | 单文件行打印「脱敏 4 处」，说明脱敏后的文本才进向量库 | ☐ |
| C5 | **演示结束后清理**：删除拷入的示例 md，`python ingest.py --rebuild` | 块数回到只有真实语料的值 | ☐ |

> C5 不要跳过。示例是虚构内容，长期混在真实库里会污染检索结果，
> 更糟的是演示时被误读成真实条文。

### 阶段 D · 前端交互（人工确认，无自动断言）

| # | 操作 | 预期结果 | 勾选 |
|---|---|---|---|
| D1 | 点推荐问题 | 自动作为提问发送 | ☐ |
| D2 | 点来源卡片标题行 | 展开/收起该片段原文 | ☐ |
| D3 | 答案流式输出时 | 正文**逐字出现**（有光标动效），不是等待后一次性出现 | ☐ |
| D4 | 手机连同一 Wi-Fi 访问 `http://<电脑IP>:8000` | 布局正常，左上角 ☰ 可开合侧栏抽屉 | ☐ |
| D5 | 未配 `LLM_API_KEY` 时启动 | 徽标显示「仅检索模式」，提问返回**原文片段而非报错** | ☐ |
| D6 | 桌面端打开（窗口 ≥900px） | 左栏出现统计卡（总片段数 + 五类分布条）与分类树，右栏为问答区 | ☐ |
| D7 | 点侧栏分类项再提问 | 顶栏副标题变为「筛选：XX」，来源只出现该分类文档 | ☐ |
| D8 | 连续问两个问题后刷新页面 | 侧栏「历史会话」记录仍在，点击可完整还原该次问答 | ☐ |
| D9 | 点答案里的 `[1]` 标注 | 对应来源卡片展开并高亮闪烁 | ☐ |
| D10 | 提问「承接查验包括哪些内容？」 | 答案中的编号列表 / 表格 / 加粗**按 Markdown 正常排版**，不是一坨纯文本 | ☐ |
| D11 | 提问 PDF 来源的问题（如「未来社区的验收流程分几个步骤？」）后展开来源卡片 | 显示**真实页码**（如「第 6 页」），并与该 PDF 实际内容对得上 | ☐ |
| D12 | 提问 docx 来源的问题（如「设备告警怎么配置？」）后展开来源卡片 | 显示「章节路径 · 第 N 段」（如「物业管理 > 设备告警 · 第 3 段」），**不出现「第 0 页」** | ☐ |
| D13 | 点来源卡片里的「复制路径」 | 剪贴板得到「文件相对路径 · 位置」，且卡片**不折叠**、按钮短暂变为「已复制」 | ☐ |
| D14 | 提问 md/txt 来源的问题（可拷入 `corpus_samples` 的示例） | 位置行显示「该格式无位置信息」，**不报错** | ☐ |
| D15 | 手机宽度（<900px）展开来源卡片 | 面板正常展开，长片段在面板内滚动（`max-height: 40vh`），不撑破页面 | ☐ |
| D16 | 点来源卡片行内的「原文 ›」 | 打开阅读浮层，自动滚到命中片段并高亮；副标题显示分类与片段数 | ☐ |
| D17 | 展开来源卡片后点定位串本身 | 与 D16 等效（两个入口都能打开并定位到同一处） | ☐ |
| D18 | 对 PDF 来源查看原文 | 正文按「第 N 页」分节，命中块的页标签同步高亮 | ☐ |
| D19 | 对 docx 来源查看原文 | 正文按「章节路径」分节，不出现「第 0 页」 | ☐ |
| D20 | 在浮层里滚到别处，点「回到命中处」 | 滚回命中片段并重放一次高亮 | ☐ |
| D21 | 分别按 ✕ / Esc / 点浮层空白处 | 三种方式都能关闭；关闭后对话区滚动位置不变 | ☐ |
| D22 | 打开浮层后尝试滚动底层对话 | 底层页面**不滚动**（浮层打开时锁 body 滚动） | ☐ |
| D23 | 手机宽度打开浮层 | 铺满全屏，正文可滚动，底部定位条与按钮不被裁掉 | ☐ |
| D24 | 来源指向的文档**已从库中移除**时点「原文」 | 浮层内显示错误提示（404 文案），不白屏、不静默失败 | ☐ |

### 一票否决项

以下任一项不过，视为验收不通过：

1. **B7 不通过** —— 模型对语料外的问题编造答案。这是知识库问答的红线。
2. **引用错位** —— 答案里的 `[1]` 与来源卡片第 1 条不是同一份文档。
3. **密钥泄露** —— `.env` 或 `chroma_db/` 出现在 `git status` 里。
4. **C5 未执行** —— 虚构示例留在真实库中。

## 技术亮点速览

1. **筛选存储**：脱敏（正则 + 敏感词 + 金额开关）、3-gram Jaccard 跨文档去重、目录自动分类打标。
2. **中文友好分块**：先切句、再贪心装箱、尾部回取重叠，不切断句子；超长单句硬切兜底。
3. **双阈值**：mock 与真实向量模式各用各的相似度阈值，避免离线检索为空。
4. **惰性导出**：`rag/__init__.py` 用 `__getattr__` 按需加载，冷启动不拉起重依赖。
5. **SSE 流式**：来源列表在流式前用 `format_context` 定好，保证 `[1][2]` 编号与前端来源严格对应。
6. **引用可定位**：docx 按文档流遍历 + 标题栈产出「章节路径 + 章节内段号」，PDF 逐页保留真实页号，
   分块时用字符偏移二分反查；`_build_anchored` 断言规整一致性，不一致自动降级为无位置信息。
7. **引用可点开**：`/api/doc` 按来源取回入库片段，前端拼成阅读视图并三级回退定位
   （`chunk_id` → `chunk_index` → 「页 + 章节 + 段号」全等），历史会话没有 `chunk_id` 也能定位；
   远距离滚动直接跳（483 块的大文档正文高 20 万像素，平滑滚动要飞 1.5 秒）；
   只读向量库不读原文件，展示的即已脱敏内容，不把入库时打码掉的隐私又漏出去。
