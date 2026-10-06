你是资深的 RAG 架构师，也是我的结对技术搭档。我们要从零搭建一个「未来社区 + 智慧物业知识库问答系统」。要求：能跑通、能演示、代码规范、能讲出技术亮点。

## 项目定位
一个私有知识库问答系统：
- 把物业/未来社区领域的五类文档入库：行业标准、交付流程、操作手册、交付文档、过往案例；
- 入库前做「筛选存储」（脱敏、去重、分类打标），这是本项目区别于普通 RAG demo 的核心亮点；
- 通过 RAG 支持自然语言问答，回答带来源引用；
- 前端是可交互的 H5 页面，手机端可用。

## 已锁定技术栈（除非有充分理由，否则不要更改）
- Python 3.11
- 对话模型：DeepSeek API（OpenAI 兼容，base_url=https://api.deepseek.com，model=deepseek-chat）
- 向量模型：硅基流动 SiliconFlow 的 BAAI/bge-m3（OpenAI 兼容，base_url=https://api.siliconflow.cn/v1，维度 1024）
- 统一用 openai Python SDK，靠 base_url 区分供应商，方便以后切换
- 向量库：ChromaDB（本地持久化）
- 后端：FastAPI
- 前端：原生 H5 单页（HTML + CSS + JS），移动端优先，不引入前端框架
- 密钥放 .env（提供 .env.example），绝不硬编码或提交

## 核心功能
1. 入库管道 ingest.py：解析(docx/pdf/md/txt) → 中文分块(带重叠) → 筛选(脱敏/去重/打标) → 向量化 → 写入 ChromaDB。
2. 问答后端 api.py：/api/ask（问题→向量化→检索 top-k→拼接上下文→大模型作答→返回答案+来源引用）；/api/categories（分类与数量）。
3. H5 前端 static/：聊天界面、分类筛选、推荐问题、答案下方展示引用来源与匹配度。
4. mock 向量模式：EMBED_PROVIDER=mock 时不调任何接口，用于离线跑通全链路。

## 目录结构（按此创建）
```
future-community-rag/
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── ingest.py
├── api.py
├── README.md
├── rag/
│   ├── loader.py
│   ├── chunker.py
│   ├── screener.py
│   ├── embedder.py
│   ├── store.py
│   ├── retriever.py
│   └── llm.py
├── corpus/
│   ├── 01_行业标准/
│   ├── 02_交付流程/
│   ├── 03_操作手册/
│   ├── 04_交付文档/
│   └── 05_过往案例/
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## 编码规范
- 中文注释，模块化、职责单一；
- 每个函数说明输入输出；
- 写 README（安装/配置/入库/启动/部署说明）。

## 协作方式
- 先给出实施方案和目录结构，等我确认后再写代码；
- 按阶段推进，每完成一个模块说明它做了什么、如何验证；
- 不要一次性把所有代码写完。