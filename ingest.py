"""
入库管道入口
============

职责：把 corpus/ 下的原始文档批量灌进向量库，串起完整的五步流水线：

    解析 loader → 分块 chunker → 筛选 screener → 向量化 embedder → 入库 store

用法：
    python ingest.py                      # 增量入库全部语料
    python ingest.py --rebuild            # 清空向量库后重建（语料有删改时用）
    python ingest.py --category past_case # 只入库某个分类
    python ingest.py --mask-amounts       # 额外脱敏金额（合同/报价类语料建议开）
    python ingest.py --dry-run            # 只跑解析与筛选，不写库，用于检查筛选效果

离线跑通：.env 里设 EMBED_PROVIDER=mock（配合 LLM_PROVIDER=mock 可全程不联网）。

关键设计：筛选必须按「整批语料」做，不能逐个文件做
--------------------------------------------------
    「同一份标准被复制进多个案例文档」是本项目要解决的核心脏数据类型，
    而重复的两块天然分处不同文件。若按文件逐个筛选，去重只能看到文件内部，
    跨文档重复永远发现不了（实测：两份只差业主姓名/手机号的案例文档会全部入库）。
    因此管道分成三段：
        阶段一  逐文件解析 + 分块（保留 per-file 进度与异常隔离）
        阶段二  全语料汇总后一次性脱敏 + 去重（跨文档可见）
        阶段三  按文件分组回写：向量化 + 先删旧块再写新块
    这样既拿到了全局去重，又没有丢掉逐文件的进度输出与失败隔离。
"""

import argparse
import sys
import time
from dataclasses import dataclass, field

import config
from rag.chunker import Chunk, chunk_document
from rag.embedder import get_embedder
from rag.loader import LoadedDoc, load_file
from rag.screener import screen_chunks
from rag.store import VectorStore


@dataclass
class DocStat:
    """
    单个文件的入库统计。

    输入：由 main 的阶段一、阶段三填充。
    输出字段：source/title/category_name 标识，chunks 原始块数，kept 入库块数，
             dup 被去重掉的块数，mask_total 脱敏命中处数，seconds 耗时，error 失败原因。
    """

    source: str
    title: str = ""
    category_name: str = ""
    chunks: int = 0
    kept: int = 0
    dup: int = 0
    mask_total: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class PipelineStat:
    """整批入库汇总。输入：由 main 累加；输出字段为各维度总计。"""

    docs: int = 0
    failed: int = 0
    chunks: int = 0
    kept: int = 0
    dup: int = 0
    mask_hits: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    输入：无（读 sys.argv）；输出：argparse.Namespace。
    """
    parser = argparse.ArgumentParser(
        description="未来社区知识库入库管道",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rebuild", action="store_true", help="清空向量库后全量重建")
    parser.add_argument(
        "--category",
        default="all",
        choices=["all"] + [c["id"] for c in config.CATEGORIES],
        help="只入库指定分类，缺省 all（全部分类）",
    )
    parser.add_argument("--mask-amounts", action="store_true", help="额外脱敏金额（合同、报价类语料建议开启）")
    parser.add_argument("--dry-run", action="store_true", help="只解析与筛选，不写向量库")
    return parser.parse_args()


def collect_files(category: str) -> list:
    """
    收集待入库文件。

    输入：分类 id 或 "all"；输出：按路径排序的受支持文件列表。
    说明：按目录定向收集而非全量扫描后再过滤，避免大语料下白读一堆文件。
          不支持的格式会汇总打印提示 —— 静默跳过会让用户以为「文件放进去就能用」，
          实际提问时只看到「未收录」，无从排查到底是没入库还是没命中。
    """
    targets = config.CATEGORIES if category == "all" else [c for c in config.CATEGORIES if c["id"] == category]
    files: list = []
    skipped: dict[str, list[str]] = {}      # 后缀 -> 该后缀下的文件名列表

    for item in targets:
        folder = config.CORPUS_DIR / item["dir"]
        if not folder.exists():
            print(f"[ingest] 目录不存在，跳过：{folder}")
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue                    # 忽略隐藏文件（.gitkeep 之类）
            if path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
                files.append(path)
            else:
                skipped.setdefault(path.suffix.lower() or "(无后缀)", []).append(path.name)

    if skipped:
        total = sum(len(names) for names in skipped.values())
        detail = "、".join(f"{ext} ×{len(names)}" for ext, names in sorted(skipped.items()))
        print(f"[提示] {total} 个文件格式不支持，已跳过：{detail}")
        for ext, names in sorted(skipped.items()):
            for name in names[:3]:
                print(f"         - {name}")
            if len(names) > 3:
                print(f"         - …同后缀另有 {len(names) - 3} 个")
        print(f"       支持的后缀：{sorted(config.SUPPORTED_EXTENSIONS)}；"
              f"如需入库请先转成上述格式（.doc → .docx，.pptx → .pdf 等）")

    return sorted(files)


# ------------------------------------------------------------------ 阶段一
def load_and_chunk(files: list) -> tuple[list[DocStat], dict[str, list[Chunk]]]:
    """
    阶段一：逐文件解析并分块。

    输入：文件路径列表；输出：(按文件顺序的 DocStat 列表, {来源: 块列表})。
    说明：单个文件失败只记录错误并跳过，不中断整批——语料里难免有损坏的 docx/pdf
          或抽不出文字的扫描件。
    """
    stats: list[DocStat] = []
    chunks_by_source: dict[str, list[Chunk]] = {}

    for path in files:
        stat = DocStat(source=path.name)
        try:
            doc: LoadedDoc = load_file(path, config.CORPUS_DIR)
            if doc.is_empty:
                raise ValueError("正文为空（可能是扫描版 PDF）")
            chunks = chunk_document(doc)
            if not chunks:
                raise ValueError(f"分块后没有可用内容（均短于 {config.CHUNK_MIN_LENGTH} 字）")
        except Exception as exc:                 # noqa: BLE001 - 单文件失败不影响整批
            stat.error = str(exc)
            stats.append(stat)
            continue

        stat.source = doc.source
        stat.title = doc.title
        stat.category_name = config.CATEGORY_NAME.get(doc.category, doc.category)
        stat.chunks = len(chunks)
        stats.append(stat)
        chunks_by_source[doc.source] = chunks

    return stats, chunks_by_source


# ------------------------------------------------------------------ 阶段三
def embed_and_store(
    chunks: list[Chunk],
    store: VectorStore,
    embedder,
) -> None:
    """
    阶段三：向量化并写入向量库。

    输入：某文件保留的块列表、向量库、向量化器；输出：无。
    说明：写前先按来源删除旧块——文档改动会产生新 chunk_id，upsert 覆盖不掉旧块，
          不先删就会残留历史版本污染检索结果。
    """
    if not chunks:
        return
    vectors = embedder.embed_texts([c.embed_text for c in chunks])
    store.delete_by_source(chunks[0].source)
    store.add(chunks, vectors)


# ------------------------------------------------------------------ 输出
def format_masks(masks: dict[str, int]) -> str:
    """
    脱敏明细转展示串。

    输入：{类型: 次数}；输出：形如 "手机号×2、敏感词×1" 的字符串，空则返回 "无"。
    """
    return "、".join(f"{k}×{v}" for k, v in masks.items()) or "无"


def print_stat(stat: DocStat) -> None:
    """
    打印单文件处理结果。

    输入：DocStat；输出：无（打印一行）。
    """
    if stat.error:
        print(f"  [失败] {stat.source}：{stat.error}")
        return
    print(
        f"  [完成] {stat.source}"
        f"  →  块 {stat.chunks} / 入库 {stat.kept} / 去重 {stat.dup}"
        f"  |  脱敏 {stat.mask_total} 处  |  {stat.seconds:.1f}s"
    )


def print_summary(pipeline: PipelineStat, store: VectorStore, dry_run: bool) -> None:
    """
    打印整批汇总与库内分类分布。

    输入：汇总统计、向量库、是否 dry-run；输出：无（打印多行）。
    """
    print("\n" + "=" * 64)
    print("入库汇总")
    print("=" * 64)
    print(f"文档      : 成功 {pipeline.docs} 篇，失败 {pipeline.failed} 篇")
    print(f"分块      : 原始 {pipeline.chunks} 块 → 入库 {pipeline.kept} 块（跨文档去重 {pipeline.dup} 块）")
    print(f"脱敏命中  : {format_masks(pipeline.mask_hits)}")
    print(f"总耗时    : {pipeline.seconds:.1f}s")

    if dry_run:
        print("模式      : dry-run（未写入向量库）")
        print("=" * 64)
        return

    print("-" * 64)
    print("库内分类分布")
    counts = store.category_counts()
    total = store.count()
    for item in config.CATEGORIES:
        count = counts.get(item["id"], 0)
        bar = "█" * min(40, count // max(1, total // 40)) if total else ""
        print(f"  {item['name']:<8} {count:>6} 块  {bar}")
    unknown = {k: v for k, v in counts.items() if k not in config.CATEGORY_NAME}
    if unknown:
        print(f"  {'未分类':<8} {sum(unknown.values()):>6} 块")
    print(f"  {'合计':<8} {total:>6} 块")
    print("=" * 64)


# ------------------------------------------------------------------ 主流程
def main() -> int:
    """
    管道主流程。

    输入：命令行参数；输出：进程退出码（0 成功，1 前置检查失败）。
    """
    args = parse_args()
    embedder = get_embedder()
    store = VectorStore()

    # ---------------- 前置检查：把配置问题挡在耗时操作之前 ----------------
    print("=" * 64)
    print("未来社区知识库 · 入库管道")
    print("=" * 64)
    print(f"语料目录  : {config.CORPUS_DIR}")
    print(f"向量库    : {store.persist_dir}（集合 {store.collection_name}）")
    print(f"向量模型  : {embedder.model} dim={embedder.dim} provider={embedder.provider}")
    print(f"分块参数  : size={config.CHUNK_SIZE} overlap={config.CHUNK_OVERLAP} min={config.CHUNK_MIN_LENGTH}")
    print(f"金额脱敏  : {'开启' if args.mask_amounts else '关闭'}")
    print(f"分类范围  : {args.category}")
    print("-" * 64)

    if not embedder.is_mock and not embedder.api_key:
        print("[错误] 未配置 EMBED_API_KEY。请在 .env 中填写，或设置 EMBED_PROVIDER=mock 离线跑通。")
        return 1

    files = collect_files(args.category)
    if not files:
        print("[提示] 没有找到待入库的文档。请把 docx/pdf/md/txt 放进 corpus/ 对应分类目录。")
        return 0

    # ---------------- 重建 / 增量 ----------------
    if args.rebuild and not args.dry_run:
        before = store.count()
        store.reset()
        print(f"[重建] 已清空向量库（原有 {before} 块）")
    elif store.count() and not args.dry_run:
        print(f"[提示] 当前库内已有 {store.count()} 块，本次为增量入库；如需彻底重建请加 --rebuild")

    batch_started = time.perf_counter()

    # ---------------- 阶段一：逐文件解析 + 分块 ----------------
    print(f"待处理文档：{len(files)} 篇")
    phase_started = time.perf_counter()
    stats, chunks_by_source = load_and_chunk(files)
    ok_stats = [s for s in stats if not s.error]
    all_chunks = [chunk for stat in ok_stats for chunk in chunks_by_source[stat.source]]
    print(
        f"[1/3 解析分块] {len(ok_stats)} 篇成功 / {len(stats) - len(ok_stats)} 篇失败"
        f"  →  {len(all_chunks)} 块  （{time.perf_counter() - phase_started:.1f}s）"
    )

    # ---------------- 阶段二：全语料一次性脱敏 + 跨文档去重 ----------------
    if all_chunks:
        phase_started = time.perf_counter()
        screened = screen_chunks(all_chunks, enable_amount=args.mask_amounts)
        print(f"[2/3 筛选]     {screened.summary()}")
    else:
        screened = None
        print("[2/3 筛选]     无可处理内容，跳过")

    # ---------------- 阶段三：按文件回写向量库 ----------------
    kept_by_source: dict[str, list[Chunk]] = {}
    if screened:
        for chunk in screened.chunks:
            kept_by_source.setdefault(chunk.source, []).append(chunk)

    print(f"[3/3 入库]     {'（dry-run 跳过）' if args.dry_run else '写入中…'}")
    for stat in stats:
        # 阶段一就失败的文件在这里一并报出，避免「失败无声」
        if stat.error:
            print_stat(stat)
            continue

        stat_started = time.perf_counter()
        own_chunks = chunks_by_source[stat.source]
        kept = kept_by_source.get(stat.source, [])

        stat.kept = len(kept)
        stat.dup = stat.chunks - stat.kept
        # 脱敏处数取「本文件全部块」：命中信息在脱敏阶段就写进了块对象，
        # 即便该块随后因去重被丢弃，它身上的脱敏工作量也应计入，否则统计偏小
        stat.mask_total = sum(chunk.mask_hits for chunk in own_chunks)

        try:
            if not args.dry_run:
                embed_and_store(kept, store, embedder)
        except Exception as exc:                 # noqa: BLE001 - 单文件失败不影响整批
            stat.error = str(exc)
        stat.seconds = time.perf_counter() - stat_started
        print_stat(stat)

    # ---------------- 汇总 ----------------
    pipeline = PipelineStat()
    pipeline.docs = sum(1 for s in stats if not s.error)
    pipeline.failed = sum(1 for s in stats if s.error)
    pipeline.chunks = sum(s.chunks for s in stats if not s.error)
    pipeline.kept = sum(s.kept for s in stats if not s.error)
    # 去重与脱敏的权威口径取筛选器自身的统计（它看到的是整批语料）
    if screened:
        pipeline.dup = screened.dup_exact + screened.dup_near
        pipeline.mask_hits = dict(screened.mask_hits)
    pipeline.seconds = time.perf_counter() - batch_started

    print_summary(pipeline, store, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
