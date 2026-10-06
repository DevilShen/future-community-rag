"""一键示例评测：把 corpus_samples 入到隔离向量库，连跑检索 + 生成评测。

用途：在不碰真实库（corpus/ + chroma_db/future_community_kb）的前提下，
用 5 份虚构示例语料跑出「有意义的基线」，验证评测方法论与全链路。

隔离方式：独立语料 corpus_samples + 独立向量库 eval_chroma + 独立集合 eval_samples_kb。

用法：
    python run_sample_eval.py                # 重建隔离库并评测（默认）
    python run_sample_eval.py --no-rebuild   # 复用上次已入库结果，只评测
"""

import sys

# 强制 stdout 用 UTF-8 输出：避免 Windows 默认 GBK 控制台打印中文/特殊符号时报错。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from pathlib import Path

# 项目根 = 本文件所在目录的上一级（eval/ 的上一级）
PROJECT = Path(__file__).resolve().parents[1]
EVAL = Path(__file__).resolve().parent
# 把「项目根」和「eval/」都加进 sys.path：
#   - 项目根：用于 import config 和 rag.*
#   - eval/  ：用于 import retrieval_eval / generation_eval
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(EVAL))

import config
from rag.loader import load_dir
from rag.chunker import chunk_document
from rag.screener import screen_chunks
from rag.embedder import get_embedder
from rag.store import VectorStore
from rag.retriever import Retriever
from rag.llm import get_llm

import retrieval_eval
import generation_eval

# 隔离三件套：独立语料、独立向量库目录、独立集合名——绝不碰真实库
CORPUS = config.BASE_DIR / "corpus_samples"
CHROMA = config.BASE_DIR / "eval_chroma"
COLLECTION = "eval_samples_kb"


def ingest(rebuild=True):
    """把 corpus_samples 入到隔离向量库，返回 (store, embedder)。

    流水线四步（与项目 ingest.py 一致，只是换成隔离的语料与向量库）：
        解析 load_dir → 分块 chunk_document → 筛选 screen_chunks → 向量化 + 入库
    """
    store = VectorStore(persist_dir=str(CHROMA), collection_name=COLLECTION)
    if rebuild:
        store.reset()          # 重建：清空隔离集合，避免旧数据残留

    embedder = get_embedder()  # 读 .env 决定 mock / siliconflow

    # 解析语料；过滤掉「uncategorized」，即排除 corpus_samples 顶层的 README.md
    # （README 是说明文件不是知识正文，混入会污染检索结果）
    docs = [d for d in load_dir(CORPUS) if d.category != 'uncategorized']
    if not docs:
        print(f"[错误] corpus_samples 下没有可入库的文档：{CORPUS}")
        raise SystemExit(1)

    # 逐篇分块，拼成一个总列表（跨文档去重需要整批做，见项目 screener 说明）
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    # 筛选：脱敏 + 去重 + 打标（enable_amount=False 不额外打码金额）
    screened = screen_chunks(all_chunks, enable_amount=False)

    # 向量化：用每个块的 embed_text（分类+标题前缀+正文），与项目入库一致
    vectors = embedder.embed_texts([c.embed_text for c in screened.chunks])
    store.add(screened.chunks, vectors)

    print(f"[入库] corpus_samples：{len(docs)} 篇 → {len(screened.chunks)} 块")
    print(f"[脱敏] 命中 {screened.mask_hits}")
    print(f"[隔离] 向量库 {CHROMA.name} / 集合 {COLLECTION}（真实库未受影响）")
    print(f"[向量] provider={embedder.provider} model={embedder.model} dim={embedder.dim}")
    return store, embedder


def main():
    import argparse
    p = argparse.ArgumentParser(description="隔离示例评测")
    p.add_argument("--no-rebuild", action="store_true", help="复用上次已入库结果")
    args = p.parse_args()

    # 1) 入库（默认重建）
    store, embedder = ingest(rebuild=not args.no_rebuild)

    # 2) 用隔离库构造检索器 + 取全局 LLM 单例 + 读题
    retriever = Retriever(store=store, embedder=embedder)
    llm = get_llm()
    questions = retrieval_eval.load_questions(EVAL / "questions.jsonl")

    # 3) 先跑检索评测，再跑生成评测
    print("\n" + "#" * 64)
    print("# 一、检索评测")
    print("#" * 64)
    retrieval_eval.report(retrieval_eval.evaluate(questions, retriever, top_k=5, category_filter=None))

    print("\n" + "#" * 64)
    print("# 二、生成评测")
    print("#" * 64)
    # 提示当前生成模式：mock / 仅检索 时答案非模型生成，结果只作结构自检
    if config.is_mock_llm():
        print("[提示] 当前 LLM_PROVIDER=mock，答案为模板，生成评测结果仅作结构自检。")
    elif not config.LLM_API_KEY:
        print("[提示] 未配置 LLM_API_KEY，当前为「仅检索」模式，答案非模型生成。")
    generation_eval.report(generation_eval.evaluate(questions, retriever, llm, config.MASK_TOKEN))


if __name__ == "__main__":
    main()