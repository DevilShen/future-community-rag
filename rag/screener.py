"""
筛选存储模块（本项目的核心差异化环节）
=====================================

职责：向量化之前先做一次「体检」，把不能入库、不必入库的内容挡在向量库外：
    1. 脱敏 —— 结构化隐私正则打码（手机号/身份证/银行卡/固话/邮箱）+ 可配置敏感词
               + 可选金额打码；
    2. 去重 —— 内容哈希做精确去重，再按相似度阈值做近似去重；
    3. 打标 —— 补齐分类/来源/标题元数据，回填脱敏命中次数。

为什么把这个环节独立成模块：
    普通 RAG demo 直接把文档灌进向量库，会导致①业主手机号被原样检索出来、
    ②同一份标准在多个文档里重复出现，挤占 top-k 名额、成本翻倍。
    这两类问题在向量库里是**洗不掉的**（脏数据已经进入索引），必须前置拦截。

处理顺序（重要）：
    脱敏 → 去重 → 打标。先脱敏再去重，因为两段只差一个手机号的文本
    脱敏后才暴露出重复；顺序反过来会漏判。

对外接口：
    desensitize(text)        -> (脱敏后文本, {类型: 命中次数})
    content_hash(text)       -> str
    screen_chunks(chunks)    -> ScreenResult
"""

import hashlib
import re
import time
import zlib
from dataclasses import dataclass, field

import config
from rag.chunker import Chunk

# 参与相似度计算的 n-gram 长度，中文用 3-gram 兼顾区分度与容错
_SHINGLE_N = 3

# ---------------- 近似去重的 MinHash 参数 ----------------
# 签名长度 16 = 4 带 × 每带 4 行。分带的目的是把「两两比较」换成「按带分桶后只在桶内比较」：
# Jaccard 越高的两块越可能至少共享一带，从而被放到同一候选桶里。
_MINHASH_SIZE = 16
_BAND_ROWS = 4
_NUM_BANDS = _MINHASH_SIZE // _BAND_ROWS
# 单个候选桶的比较上限，防止模板化文本（如全篇重复的免责声明）让桶退化成 O(n²)
_MAX_CANDIDATE_BUCKET = 500
# 取模用的大素数，配合线性哈希族 (a*x+b) mod P 构造多个独立哈希函数
_HASH_PRIME = (1 << 31) - 1
# 线性哈希族参数，用固定公式生成而非随机数，保证每次运行结果可复现
_HASH_PARAMS: list[tuple[int, int]] = [
    (((i + 1) * 2654435761) % _HASH_PRIME | 1, ((i + 1) * 40503 + 7) % _HASH_PRIME)
    for i in range(_MINHASH_SIZE)
]


@dataclass
class ScreenResult:
    """
    筛选结果汇总。

    输入：由 screen_chunks 构造。
    输出字段：
        chunks        : 通过筛选、可入库的 Chunk 列表
        total         : 输入块总数
        kept          : 保留数量
        dup_exact     : 精确哈希去重掉的块数
        dup_near      : 近似去重掉的块数
        mask_hits     : 各类脱敏命中的总次数，如 {"手机号": 3, "敏感词": 1}
        elapsed_ms    : 筛选耗时（毫秒），用于验证这个环节的开销可接受
    """

    chunks: list[Chunk] = field(default_factory=list)
    total: int = 0
    kept: int = 0
    dup_exact: int = 0
    dup_near: int = 0
    mask_hits: dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def summary(self) -> str:
        """输出一行人类可读的统计，供入库日志打印。"""
        masks = "、".join(f"{k}×{v}" for k, v in self.mask_hits.items()) or "无"
        return (
            f"共 {self.total} 块 → 保留 {self.kept} 块"
            f"（精确去重 {self.dup_exact}，近似去重 {self.dup_near}）"
            f"｜脱敏命中：{masks}｜耗时 {self.elapsed_ms:.0f}ms"
        )


# ------------------------------------------------------------------ 脱敏
def _build_patterns(enable_amount: bool) -> list[tuple[str, re.Pattern[str]]]:
    """
    组装脱敏规则。

    输入：是否启用金额脱敏；输出：[(规则名, 编译后的正则), ...]。
    顺序即优先级：config 里已按「长模式在前」排列，避免短规则先吃掉长串的一部分。
    """
    rules = list(config.DESENSITIZE_PATTERNS)
    if enable_amount:
        rules.append(("金额", config.AMOUNT_PATTERN))
    return [(name, re.compile(pattern)) for name, pattern in rules]


def desensitize(
    text: str,
    extra_words: list[str] | None = None,
    enable_amount: bool | None = None,
    mask_token: str | None = None,
) -> tuple[str, dict[str, int]]:
    """
    对单段文本脱敏。

    输入：
        text          : 待脱敏文本
        extra_words   : 额外敏感词，缺省用 config.SENSITIVE_WORDS（来自 .env，可放客户名）
        enable_amount : 是否脱敏金额，缺省用 config.MASK_AMOUNT
        mask_token    : 替换占位符，缺省用 config.MASK_TOKEN
    输出：(脱敏后文本, {规则名: 命中次数})；无命中时原样返回，不做任何改写。
    """
    words = config.SENSITIVE_WORDS if extra_words is None else extra_words
    enable_amount = config.MASK_AMOUNT if enable_amount is None else enable_amount
    mask_token = mask_token or config.MASK_TOKEN

    hits: dict[str, int] = {}
    result = text

    # 1) 结构化隐私：正则整体替换，一条规则一次扫完
    for name, pattern in _build_patterns(enable_amount):
        result, count = pattern.subn(mask_token, result)
        if count:
            hits[name] = hits.get(name, 0) + count

    # 2) 非结构化敏感词：逐个字面量替换，长词优先，避免「张三丰」被「张三」误伤
    for word in sorted(words, key=len, reverse=True):
        if not word:
            continue
        # 用 re.escape 防止敏感词里的 . * ( ) 被当成正则元字符
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        result, count = pattern.subn(mask_token, result)
        if count:
            hits["敏感词"] = hits.get("敏感词", 0) + count

    return result, hits


# ------------------------------------------------------------------ 去重
def content_hash(text: str) -> str:
    """
    计算内容指纹。

    输入：文本；输出：32 位十六进制 md5。
    归一化策略：去掉全部空白字符 + 英文转小写。
    理由：中文语料里同一段内容的差异往往只是排版产生的空格/换行，
         不归一化会漏掉大量真重复；本场景下「去空格后相同」即可判定等价。
    """
    normalized = re.sub(r"\s+", "", text).lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _shingles(text: str) -> set[str]:
    """
    文本转 3-gram 集合。

    输入：文本；输出：去空白后的相邻 3 字符片段集合，用于快速估算相似度。
    """
    compact = re.sub(r"\s+", "", text)
    if len(compact) <= _SHINGLE_N:
        return {compact} if compact else set()
    return {compact[i:i + _SHINGLE_N] for i in range(len(compact) - _SHINGLE_N + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard 相似度。输入：两个集合；输出：0~1 的交并比。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _minhash_signature(text: str) -> tuple[int, ...] | None:
    """
    计算 MinHash 签名。

    输入：文本；输出：长度 _MINHASH_SIZE 的签名，文本过短无 shingle 时返回 None。

    原理：对 shingle 集合应用 16 个线性哈希函数，各取最小值。两个集合的签名在
    第 i 位上相等的概率恰好等于它们的 Jaccard 相似度，于是可以用「有没有若干位
    相同」快速筛出「可能重复」的候选对，避免全量两两比较。
    用 zlib.crc32 而非内置 hash()：内置 hash 每个进程随机加盐，跨进程结果不稳定。
    """
    shingles = _shingles(text)
    if not shingles:
        return None
    base = [zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF for s in shingles]
    return tuple(
        min((a * value + b) % _HASH_PRIME for value in base)
        for a, b in _HASH_PARAMS
    )


def _dedupe_exact(chunks: list[Chunk]) -> tuple[list[Chunk], int]:
    """
    精确去重（内容哈希）。

    输入：Chunk 列表；输出：(保留列表, 剔除数量)。首次出现的保留，后续相同指纹丢弃。
    """
    seen: dict[str, Chunk] = {}
    kept: list[Chunk] = []
    for chunk in chunks:
        fingerprint = content_hash(chunk.text)
        if fingerprint in seen:
            continue
        seen[fingerprint] = chunk
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


def _dedupe_near(
    chunks: list[Chunk],
    threshold: float,
) -> tuple[list[Chunk], int]:
    """
    近似去重（MinHash + LSH 分带候选 + Jaccard 校验）。

    输入：已精确去重的 Chunk 列表、相似度阈值（0~1）。
    输出：(保留列表, 剔除数量)，保留先出现的那一条。

    三步：
        1. 每块算 16 维 MinHash 签名；
        2. 签名切成 4 带，带内容相同的块进入同一候选桶（LSH，把 O(n²) 降成近似 O(n)）；
        3. 桶内两两算真实 Jaccard，达到阈值就丢弃后出现的那条。
    第 3 步必须做：LSH 只是「可能相似」的粗筛，直接按桶判重会误杀。
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    # 签名 + 分带建桶
    buckets: dict[tuple[int, tuple[int, ...]], list[str]] = {}
    for chunk in chunks:
        signature = _minhash_signature(chunk.text)
        if signature is None:
            continue
        for band in range(_NUM_BANDS):
            start = band * _BAND_ROWS
            key = (band, signature[start:start + _BAND_ROWS])
            buckets.setdefault(key, []).append(chunk.chunk_id)

    dropped: set[str] = set()
    compared: set[tuple[str, str]] = set()      # 一对块可能共享多带，去重避免重复比较
    shingle_cache: dict[str, set[str]] = {}

    def shingles_of(chunk_id: str) -> set[str]:
        """惰性计算并缓存 shingle，只有进入候选对的块才付这个成本。"""
        if chunk_id not in shingle_cache:
            shingle_cache[chunk_id] = _shingles(by_id[chunk_id].text)
        return shingle_cache[chunk_id]

    for ids in buckets.values():
        if len(ids) < 2:
            continue
        if len(ids) > _MAX_CANDIDATE_BUCKET:
            print(f"[screener] 候选桶过大（{len(ids)} 块），跳过该桶的近似去重；建议调大 _MINHASH_SIZE")
            continue
        for i, first in enumerate(ids):
            if first in dropped:
                continue
            for second in ids[i + 1:]:
                if second in dropped or first == second:
                    continue
                pair = (first, second) if first < second else (second, first)
                if pair in compared:
                    continue
                compared.add(pair)
                if _jaccard(shingles_of(first), shingles_of(second)) >= threshold:
                    dropped.add(second)          # 保留先出现的
    return [c for c in chunks if c.chunk_id not in dropped], len(dropped)


# ------------------------------------------------------------------ 打标
def _enrich(chunk: Chunk) -> Chunk:
    """
    补齐元数据。

    输入：Chunk；输出：原地补全后的同一个 Chunk。
    补齐内容：分类中文名（防上游漏填）、标题为空时退回来源文件名。
    """
    chunk.category_name = config.CATEGORY_NAME.get(chunk.category, chunk.category)
    if not chunk.title:
        chunk.title = chunk.source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    # 脱敏改动了正文，元数据前缀需要跟着重建，否则向量化用的还是旧文本
    chunk.embed_text = f"【{chunk.category_name}】{chunk.title}\n{chunk.text}"
    return chunk


# ------------------------------------------------------------------ 主流程
def screen_chunks(
    chunks: list[Chunk],
    enable_dedup: bool = True,
    enable_amount: bool | None = None,
    extra_words: list[str] | None = None,
) -> ScreenResult:
    """
    对分块结果执行完整筛选流程。

    输入：
        chunks        : chunker 产出的 Chunk 列表
        enable_dedup  : 是否开启去重（精确 + 近似）
        enable_amount : 是否脱敏金额，缺省读 config.MASK_AMOUNT
        extra_words   : 额外敏感词，缺省读 config.SENSITIVE_WORDS
    输出：ScreenResult，其中 .chunks 可直接送去向量化。

    统计口径提醒：mask_hits 统计的是「块内命中次数」。相邻块有重叠，同一处手机号
    若落在重叠区会被计两次，因此该数字是脱敏工作量而非原文出现次数。
    """
    started = time.perf_counter()
    result = ScreenResult(total=len(chunks))

    # 1) 脱敏：逐块处理，累计命中次数并回填到块元数据上
    for chunk in chunks:
        masked_text, hits = desensitize(
            chunk.text, extra_words=extra_words, enable_amount=enable_amount
        )
        if hits:
            chunk.text = masked_text
            chunk.mask_hits = sum(hits.values())
            for name, count in hits.items():
                result.mask_hits[name] = result.mask_hits.get(name, 0) + count

    # 2) 去重：先精确后近似
    kept = chunks
    if enable_dedup:
        kept, result.dup_exact = _dedupe_exact(kept)
        kept, result.dup_near = _dedupe_near(kept, config.DEDUP_SIMILARITY_THRESHOLD)

    # 3) 打标：脱敏改过正文，必须在最后重建 embed_text
    result.chunks = [_enrich(c) for c in kept]
    result.kept = len(result.chunks)
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result


if __name__ == "__main__":
    # 自检：python -m rag.screener
    from rag.loader import LoadedDoc
    from rag.chunker import chunk_document

    demo = LoadedDoc(
        text=(
            "业主王先生（手机号 13812345678，身份证 110101199003071234，邮箱 wang@example.com）"
            "报修厨房漏水，合同金额 12,000 元，内部示例公司需在 30 分钟内响应。\n"
        ) * 3,
        source="05_过往案例/示例案例.md",
        title="示例案例",
        category="past_case",
    )
    chunks = chunk_document(demo, max_chars=120, overlap=20)
    print(f"分块后：{len(chunks)} 块")
    res = screen_chunks(chunks)
    print(res.summary())
    for c in res.chunks:
        print(f"  [{c.chunk_id}] 命中{c.mask_hits} {c.text[:80]}")
