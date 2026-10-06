"""
全局配置模块
============

职责：集中读取 .env，把散落的配置收敛成一个可 import 的对象，供
      ingest.py（入库管道）、api.py（问答后端）与 rag/* 各模块复用。

约定：
    1. 所有可变参数一律走环境变量，代码里不出现魔法数字；
    2. 每个配置项都提供合理默认值，缺 .env 也能启动（mock 模式可离线跑通）；
    3. 密钥类字段只从环境变量读取，仓库内永不出现真实值。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- 基础路径
# 项目根目录（本文件所在目录），其余路径都以它为基准，保证在任意 CWD 下都能跑
BASE_DIR = Path(__file__).resolve().parent

# 加载 .env（override=True 表示 .env 覆盖同名系统环境变量，便于本地调试）
load_dotenv(BASE_DIR / ".env", override=True)


# ---------------------------------------------------------------- 读取工具
def _get_str(key: str, default: str = "") -> str:
    """读取字符串配置。输入：键名、默认值；输出：去空格后的字符串。"""
    return os.getenv(key, default).strip()


def _get_int(key: str, default: int) -> int:
    """读取整数配置。输入：键名、默认值；输出：int（非法值回退默认值）。"""
    try:
        return int(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


def _get_float(key: str, default: float) -> float:
    """读取浮点配置。输入：键名、默认值；输出：float（非法值回退默认值）。"""
    try:
        return float(os.getenv(key, "").strip())
    except (TypeError, ValueError):
        return default


def _get_list(key: str, default: str = "") -> list[str]:
    """读取逗号分隔列表。输入：键名、默认逗号串；输出：去空去重后的字符串列表。"""
    raw = os.getenv(key, default)
    items = [item.strip() for item in raw.split(",")]
    # 保序去重，避免 .env 里手滑写重复
    return list(dict.fromkeys(item for item in items if item))


# ---------------------------------------------------------------- 目录
CORPUS_DIR = BASE_DIR / "corpus"        # 原始文档目录，五个分类子目录
CHROMA_DIR = BASE_DIR / _get_str("CHROMA_DIR", "chroma_db")   # 向量库持久化目录
STATIC_DIR = BASE_DIR / "static"        # H5 前端静态文件目录

# ---------------------------------------------------------------- 对话模型
# DeepSeek 走 OpenAI 兼容协议：同一套 openai SDK，只换 base_url / model
# LLM_PROVIDER=deepseek 调真实接口；=mock 时用检索到的片段拼一个模板化回答，
# 让「EMBED_PROVIDER=mock + LLM_PROVIDER=mock」组合能在完全离线状态下跑通全链路
LLM_PROVIDER = _get_str("LLM_PROVIDER", "deepseek").lower()
LLM_API_KEY = _get_str("LLM_API_KEY")
LLM_BASE_URL = _get_str("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = _get_str("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = _get_float("LLM_TEMPERATURE", 0.2)   # 问答场景偏严谨
# 单次生成上限。推理型模型（如 deepseek-v4-flash）会先把思考写进 reasoning_content，
# 这部分同样计入 max_tokens；预算给小了会 finish_reason=length 且正文为空，
# 所以这里按「思考 + 正文」留足，默认 2048 而非 1024。
LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 2048)

# 附加请求头，形如 "User-Agent: claude-cli/1.0.0"。留空则不加任何自定义头。
# 用途：公司中转站按客户端身份（User-Agent）放行部分模型，非白名单客户端回 403
# agent_not_allowed。⚠️ 填了相当于伪装成白名单客户端，属绕过访问控制，需确认合规。
LLM_USER_AGENT = _get_str("LLM_USER_AGENT")
# 拼进 prompt 的上下文正文总字符上限，超出按检索排名截断，防止超长文档撑爆输入
CONTEXT_MAX_CHARS = _get_int("CONTEXT_MAX_CHARS", 6000)

# ---------------------------------------------------------------- 向量模型
# EMBED_PROVIDER=mock 时不调任何外部接口，用哈希生成假向量，用于离线跑通全链路
EMBED_PROVIDER = _get_str("EMBED_PROVIDER", "mock").lower()
EMBED_API_KEY = _get_str("EMBED_API_KEY")
EMBED_BASE_URL = _get_str("EMBED_BASE_URL", "https://api.siliconflow.cn/v1")
EMBED_MODEL = _get_str("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DIM = _get_int("EMBED_DIM", 1024)     # 必须与模型真实输出维度一致
EMBED_BATCH_SIZE = _get_int("EMBED_BATCH_SIZE", 32)   # 每批请求的文本条数

# 向量集合名：换名字 = 换一个独立知识库，便于 A/B 对比不同切块策略
CHROMA_COLLECTION = _get_str("CHROMA_COLLECTION", "future_community_kb")

# ---------------------------------------------------------------- 中文分块
CHUNK_SIZE = _get_int("CHUNK_SIZE", 500)          # 单块最大字符数
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 80)     # 相邻块重叠字符数
CHUNK_MIN_LENGTH = _get_int("CHUNK_MIN_LENGTH", 30)   # 小于该长度的碎块丢弃

# ---------------------------------------------------------------- 检索
RETRIEVER_MODE = _get_str("RETRIEVER_MODE", "vector").lower()   # "vector" 纯向量 / "hybrid" 混合检索
TOP_K = _get_int("TOP_K", 5)                        # 送入大模型的片段数
SCORE_THRESHOLD = _get_float("SCORE_THRESHOLD", 0.3)  # 相似度下限，低于则丢弃

# mock 向量是「字符 3-gram 哈希词袋」，量级远低于真实模型：
# 实测相关问题对 200 字块的余弦约 0.07~0.11，无关问题约 0.00~0.05，
# 而 bge-m3 这类真实模型的相关片段通常在 0.5 以上。两种模式必须各用各的阈值，
# 否则离线模式下 0.3 会把所有结果过滤干净，链路看着通、实际检索不出东西。
SCORE_THRESHOLD_MOCK = _get_float("SCORE_THRESHOLD_MOCK", 0.02)

# ---------------------------------------------------------------- 筛选：脱敏
MASK_TOKEN = _get_str("MASK_TOKEN", "【已脱敏】")   # 命中后的统一替换占位符

# 结构化隐私：正则能覆盖的通用模式，按「先长后短」排列避免误伤
# （身份证 18 位优先于银行卡 16~19 位；前后加数字否定环视，防止从长串里截一段）
DESENSITIZE_PATTERNS: list[tuple[str, str]] = [
    ("身份证号", r"(?<!\d)\d{17}[\dXx](?!\d)"),
    ("手机号", r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ("银行卡号", r"(?<!\d)\d{16,19}(?!\d)"),
    ("固定电话", r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)"),
    ("电子邮箱", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
]

# 非结构化敏感词：正则覆盖不到的内部人名、公司名等，从 .env 注入
SENSITIVE_WORDS: list[str] = _get_list("SENSITIVE_WORDS", "")

# 金额脱敏开关：合同金额属商业敏感信息，但行业标准里的「限额」类数字是知识本体，
# 一刀切会毁掉答案，所以做成开关，按语料性质决定是否开启
MASK_AMOUNT = _get_str("MASK_AMOUNT", "false").lower() in {"1", "true", "yes", "on"}

# 金额模式：支持 1,234.56 万元 / ￥12000 / 12万元 / 3.5亿元 等写法
AMOUNT_PATTERN = r"(?:[￥¥]\s*[\d,]+(?:\.\d+)?(?:万|亿)?元?|[\d,]+(?:\.\d+)?\s*(?:万元|亿元|元|万|亿))"

# ---------------------------------------------------------------- 筛选：去重
# 入库时两段文本相似度（3-gram Jaccard）高于该阈值即判为重复，只保留首次出现的一条。
# 默认 0.9 而非 0.95：实测「只改了业主姓名和手机号的案例复制件」在块粒度上
# Jaccard 约 0.92，阈值定到 0.95 会漏掉这类最常见的复制件；
# 而 0.9 已要求近九成三字片段完全一致，两条不同条款极少达到，误杀风险可控。
DEDUP_SIMILARITY_THRESHOLD = _get_float("DEDUP_SIMILARITY_THRESHOLD", 0.9)

# ---------------------------------------------------------------- 分类体系
# 与 corpus/ 下的五个子目录一一对应：目录名即分类来源，id 用于接口与前端筛选
CATEGORIES: list[dict[str, str]] = [
    {"id": "industry_standard", "name": "行业标准", "dir": "01_行业标准"},
    {"id": "delivery_process", "name": "交付流程", "dir": "02_交付流程"},
    {"id": "operation_manual", "name": "操作手册", "dir": "03_操作手册"},
    {"id": "delivery_doc", "name": "交付文档", "dir": "04_交付文档"},
    {"id": "past_case", "name": "过往案例", "dir": "05_过往案例"},
]

# 目录名 -> 分类 id 的反查表，入库时按文件所在目录自动打标
CATEGORY_BY_DIR: dict[str, str] = {c["dir"]: c["id"] for c in CATEGORIES}

# 分类 id -> 中文名，写元数据与前端展示用
CATEGORY_NAME: dict[str, str] = {c["id"]: c["name"] for c in CATEGORIES}

# ---------------------------------------------------------------- 服务
API_HOST = _get_str("API_HOST", "0.0.0.0")
API_PORT = _get_int("API_PORT", 8000)

# 支持解析的文档后缀，loader 与入库扫描共用
SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".md", ".txt"}


def is_mock_embed() -> bool:
    """
    判断当前是否处于离线 mock 向量模式。

    输出：True 表示不调用任何向量接口，用本地假向量跑通全链路。
    """
    return EMBED_PROVIDER == "mock"


def effective_score_threshold() -> float:
    """
    返回当前向量模式下的相似度阈值。

    输入：无；输出：mock 模式取 SCORE_THRESHOLD_MOCK，真实模型取 SCORE_THRESHOLD。
    调用方（检索/问答）应统一走这个函数，不要直接读 SCORE_THRESHOLD。
    """
    return SCORE_THRESHOLD_MOCK if is_mock_embed() else SCORE_THRESHOLD


def is_mock_llm() -> bool:
    """
    判断对话模型是否处于离线 mock 模式。

    输出：True 表示不调用大模型接口，直接用检索结果拼模板回答。
    """
    return LLM_PROVIDER == "mock"


def _mask(secret: str) -> str:
    """把密钥打码后用于日志展示，输入原始密钥，输出形如 sk-a****z 的字符串。"""
    if not secret:
        return "(未配置)"
    if len(secret) <= 8:
        return secret[0] + "****"
    return f"{secret[:4]}****{secret[-2:]}"


if __name__ == "__main__":
    # 自检入口：python config.py 打印全部生效配置，便于确认 .env 是否被正确读取
    print("=" * 56)
    print("配置自检")
    print("=" * 56)
    print(f"项目根目录      : {BASE_DIR}")
    print(f"向量库目录      : {CHROMA_DIR}")
    print("-" * 56)
    print(f"LLM 模型        : {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"LLM 密钥        : {_mask(LLM_API_KEY)}")
    print(f"LLM 伪装 UA     : {LLM_USER_AGENT or '(未设置)'}")
    print(f"LLM 生成上限    : {LLM_MAX_TOKENS} tokens")
    print(f"向量供应商      : {EMBED_PROVIDER}")
    print(f"向量模型        : {EMBED_MODEL} (dim={EMBED_DIM})")
    print(f"向量密钥        : {_mask(EMBED_API_KEY)}")
    print(f"集合名          : {CHROMA_COLLECTION}")
    print("-" * 56)
    print(f"分块            : size={CHUNK_SIZE} overlap={CHUNK_OVERLAP} min={CHUNK_MIN_LENGTH}")
    print(f"检索            : top_k={TOP_K} threshold={SCORE_THRESHOLD}")
    print(f"脱敏正则        : {[name for name, _ in DESENSITIZE_PATTERNS]}")
    print(f"敏感词          : {SENSITIVE_WORDS or '(未配置)'}")
    print(f"掩码占位符      : {MASK_TOKEN}")
    print(f"去重阈值        : {DEDUP_SIMILARITY_THRESHOLD}")
    print("-" * 56)
    print(f"分类            : {[c['name'] for c in CATEGORIES]}")
    print(f"支持后缀        : {sorted(SUPPORTED_EXTENSIONS)}")
    print(f"服务地址        : http://{API_HOST}:{API_PORT}")
    print("=" * 56)
    print(f"mock 向量模式   : {is_mock_embed()}")
