"""
问答后端服务
============

职责：把 RAG 能力暴露成 HTTP 接口，并托管 H5 前端静态资源。

对外接口：
    GET  /api/health        健康检查（含 mock 状态与库内块数）
    GET  /api/categories    分类列表与各分类块数（前端筛选栏用）
    GET  /api/doc           某文档的全部入库片段（前端「查看原文」浮层用）
    POST /api/ask           问答（SSE 流式）：先推 sources 再逐段推 delta
    GET  /                  H5 首页 index.html
    /static/*               前端静态资源

设计要点：
    1. 检索与作答复用 rag 包的 Retriever / get_llm，不在此处重写业务逻辑；
    2. /api/ask 用 SSE（text/event-stream）流式返回，事件序列固定为：
           sources（来源列表）→ delta×N（正文增量）→ done（收尾），异常时插一条 error。
       来源列表在流式开始前就确定：调用 format_context 拿到「真正入选上下文的片段」，
       保证答案里的 [1][2] 编号与前端来源列表严格一一对应，不错位；
    3. 前端由本服务同源托管，一次 uvicorn api:app 即可手机同网访问，
       但仍放开 CORS 以兼容后续前后端分离部署；
    4. /api/doc 只读向量库、不读原始文件：既避免 source 参数带来的路径穿越，
       也保证「前端看到的 = 模型看到的 = 已脱敏后的内容」。
"""

import json
import re
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag import Retriever, VectorStore, format_context, get_llm

# ---------------------------------------------------------------- 应用初始化
app = FastAPI(title="未来社区 · 智慧物业知识库", version="1.0.0")

# 私有知识库，前端大概率同源托管，但放开 CORS 便于后续前后端分离或本地联调
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /api/doc 会一次性返回整篇文档的片段（最大的一篇约 500 块 / ~250KB JSON），
# 压缩后约降到 1/4，移动端同网首开才不至于明显卡顿。minimum_size 让小的
# 响应（health / categories）保持原样，不必为几十字节多付一次压缩开销。
app.add_middleware(GZipMiddleware, minimum_size=1024)

# 托管 H5 前端静态资源（css/js 等），首页 index.html 在 / 单独返回
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


# ---------------------------------------------------------------- 进程内单例
# 与 rag 包保持一致：惰性初始化，避免冷启动就打开 ChromaDB / 校验密钥。
_retriever: Retriever | None = None
_store: VectorStore | None = None


def get_retriever() -> Retriever:
    """获取全局检索器单例。输入：无；输出：Retriever。"""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_store() -> VectorStore:
    """获取全局向量库单例。输入：无；输出：VectorStore。"""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


# ---------------------------------------------------------------- 数据结构
class AskRequest(BaseModel):
    """
    /api/ask 请求体。

    字段：
        question : 用户问题（必填，去空格后非空）
        category : 分类 id 过滤，None 或 "all" 表示全库检索
        top_k    : 召回条数，None 走 config.TOP_K
    """

    question: str
    category: str | None = None
    top_k: int | None = None


# ---------------------------------------------------------------- SSE 工具
def _sse(event: str, data: dict) -> str:
    """
    序列化一条 SSE 事件。

    输入：事件名、数据字典；输出：符合 SSE 协议的一段文本（含结尾空行）。
    ensure_ascii=False 保证中文原样输出，前端无需二次解码。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _snippet(text: str, limit: int = 140) -> str:
    """
    把片段压成单行摘要，用于前端来源列表展示。

    输入：片段正文、长度上限；输出：压缩空白后的摘要，超长截断加省略号。
    """
    compact = re.sub(r"\s+", " ", text.strip())
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _source_payload(index: int, hit) -> dict:
    """
    把一条检索命中转成前端来源对象。

    输入：1 起始的编号、RetrievedChunk；
    输出：{index,title,source,chunk_id,chunk_index,category,score,snippet,
           page,heading_path,para_index,text}。
      - score 即匹配度（0~1），前端直接用于展示；
      - snippet 是折叠态的 140 字摘要；
      - page / heading_path / para_index 是「定位」三件套，三者均无时前端显示「无位置信息」；
      - chunk_id / chunk_index 是「查看原文」浮层用来精确高亮那一个块的锚点
        （chunk_id 优先；改块策略后旧块被删时回退到 chunk_index，再回退到三者全等匹配）；
      - text 是命中片段全文，供展开态展示。上限 800 字（现有块 ≤500 字，不会触发），
        防止将来调大 CHUNK_SIZE 后随会话历史塞爆 localStorage（单域名约 5MB）。
    """
    return {
        "index": index,
        "title": hit.title,
        "source": hit.source,
        "chunk_id": hit.chunk_id,
        "chunk_index": hit.chunk_index,
        "category": hit.category_name or hit.category,
        "score": hit.score,
        "snippet": _snippet(hit.text),
        "page": hit.page,
        "heading_path": hit.heading_path,
        "para_index": hit.para_index,
        "text": hit.text[:800],
    }


# ---------------------------------------------------------------- 问答流
def _answer_stream(question: str, category: str | None, top_k: int | None) -> Iterator[str]:
    """
    /api/ask 的 SSE 生成器。

    输入：问题、分类过滤、召回条数；输出：逐段 yield SSE 文本。
    事件顺序：sources → delta×N → done；异常时 yield error 后终止。

    为什么来源在流式前定好：答案正文是流式吐出的，但来源编号要与之对应，
    必须先用 format_context 确定「真正进上下文的片段」再开始推正文。
    """
    retriever = get_retriever()
    llm = get_llm()

    try:
        hits = retriever.search(question, top_k=top_k, category=category)
        # format_context 返回 (上下文文本, 实际入选片段)，第二个值才对应 [1][2] 编号
        _, used = format_context(hits)
        sources = [_source_payload(i, hit) for i, hit in enumerate(used, start=1)]
        yield _sse("sources", {"sources": sources})

        for delta in llm.answer_stream(question, hits):
            yield _sse("delta", {"text": delta})

        yield _sse("done", {"model": llm.model, "provider": llm.provider})
    except Exception as exc:  # noqa: BLE001 - 流式已开始，异常只能以事件形式回传
        yield _sse("error", {"message": str(exc)})


# ---------------------------------------------------------------- 接口实现
@app.get("/api/health")
def health() -> dict:
    """
    健康检查。

    输入：无；输出：服务状态、mock 开关、库内块数、当前模型名。
    前端加载时调用，可据此提示「离线 mock 模式」「仅检索模式」或「知识库为空请先入库」。
    retrieval_only：想调模型但没配密钥 —— 此时问答仍可用，但返回的是检索原文而非模型归纳。
    """
    return {
        "status": "ok",
        "mock_embed": config.is_mock_embed(),
        "mock_llm": config.is_mock_llm(),
        "retrieval_only": not config.is_mock_llm() and not config.LLM_API_KEY,
        "count": get_store().count(),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/api/categories")
def categories() -> dict:
    """
    分类列表与各分类块数。

    输入：无；输出：{categories:[{id,name,count}], total}，按 config.CATEGORIES 顺序，
    保证「行业标准/交付流程/…」五个分类即使块数为 0 也稳定占位，前端筛选栏完整。
    """
    counts = get_store().category_counts()
    items = [
        {"id": c["id"], "name": c["name"], "count": counts.get(c["id"], 0)}
        for c in config.CATEGORIES
    ]
    return {"categories": items, "total": get_store().count()}


@app.get("/api/doc")
def doc(source: str = Query(..., min_length=1, description="来源相对路径")) -> dict:
    """
    取某文档入库后的全部片段，供前端「查看原文」浮层渲染阅读视图。

    输入：source —— 来源相对路径（与来源卡片里显示、可复制的那串完全一致）；
    输出：
        {source, title, category, category_name, ext,
         mask_hits,     该文档入库时脱敏命中的总次数（各块 mask_hits 求和）
         chunk_count,   片段数
         chunks: [{chunk_id, chunk_index, text, page, heading_path, para_index, mask_hits}, …]}

    说明：
        - 数据只来自向量库，不读原始文件（免路径穿越，且内容与模型所见一致）；
        - ext 由 source 后缀推出，前端据此决定按「页」还是按「章节」分组；
        - 该文档查不到任何块时返回 404，而不是空文档 —— 空文档会让前端浮层白屏，
          分不清是「没入库」还是「接口坏了」。文档被整篇去重时同样落到这里。
    """
    chunks = get_store().get_by_source(source)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"该文档未入库或已从知识库移除：{source}")

    first = chunks[0]
    return {
        "source": source,
        "title": first.title,
        "category": first.category,
        "category_name": first.category_name or first.category,
        "ext": Path(source).suffix.lower(),
        "mask_hits": sum(c.mask_hits for c in chunks),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "page": c.page,
                "heading_path": c.heading_path,
                "para_index": c.para_index,
                "mask_hits": c.mask_hits,
            }
            for c in chunks
        ],
    }


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """
    问答接口（SSE 流式）。

    输入：AskRequest（question 必填）；输出：text/event-stream 响应。
    问题为空时返回 400；其余错误以 error 事件回传，避免流中途断连。
    """
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    return StreamingResponse(
        _answer_stream(question, req.category, req.top_k),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # 经 nginx 反代时禁用缓冲，保证逐字推送
        },
    )


@app.get("/")
def index() -> FileResponse:
    """H5 首页。输入：无；输出：index.html 文件响应。"""
    return FileResponse(config.STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    # python api.py 直接启动；开发期想热重载可改命令行：uvicorn api:app --reload
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
