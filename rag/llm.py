"""
大模型作答模块
==============

职责：把检索片段拼成提示词，调用对话模型，产出「带 [1][2] 引用标注」的回答。

提示词设计的四个约束（每条都对应一类常见翻车）：
    1. 只用给定资料作答  → 防止模型拿预训练知识冒充企业内规；
    2. 结论后标 [序号]   → 让答案可溯源，这是知识库问答区别于聊天的核心；
    3. 资料不足就说不知道 → 宁可答「未收录」，也不要编一条不存在的条款；
    4. 归纳而非照抄      → 直接抛原文会让引用失去意义，用户自己看原文即可。

两条实现路径：
    LLM_PROVIDER=deepseek  调真实接口（任何 OpenAI 兼容服务同理）
    LLM_PROVIDER=mock      不联网，用检索片段拼一个模板化回答，标注 [1][2] 照旧，
                           使全链路可在离线环境下演示与回归

对外接口：
    LLMClient.answer(question, chunks)       -> Answer    一次性返回
    LLMClient.answer_stream(question, chunks) -> Iterator[str]  流式返回（供 H5 打字机效果）
    get_llm()                                -> LLMClient 单例
"""

import re
from dataclasses import dataclass, field
from typing import Iterator

import config
from rag.retriever import format_context
from rag.store import RetrievedChunk

# 系统提示词：角色 + 规则，规则顺序与上面「四个约束」一一对应
SYSTEM_PROMPT = """你是「未来社区 · 智慧物业」知识库助手，服务于物业与工程交付团队。

回答规则：
1. 只依据【参考资料】作答，不得使用资料之外的知识，不得编造条款、数字、时间。
2. 每个结论后面用方括号标注来源编号，如 [1]、[2][3]，编号必须与资料一致。
3. 如果参考资料不足以回答问题，直接说明「知识库中未收录相关内容」，并指出还需要补充哪类资料，不要猜测。
4. 用简洁的中文分点作答，先给结论再给依据；融合归纳资料，不要整段照抄原文。
5. 涉及流程、时限、标准数值时，务必与资料保持完全一致。"""

# 无检索命中时的固定回复：不调模型，既省成本又保证行为确定
NO_CONTEXT_REPLY = (
    "知识库中未收录与这个问题相关的内容。\n\n"
    "建议补充以下任一类资料后重试：行业标准、交付流程、操作手册、交付文档、过往案例。"
)


@dataclass
class Answer:
    """
    一次问答的结果。

    输入：由 LLMClient.answer 构造。
    输出字段：
        text     : 回答正文（含 [1][2] 引用标注）
        sources  : 实际进入上下文的片段，按编号顺序，供接口返回给前端做引用展示
        model    : 实际使用的模型名
        provider : 供应商标识（deepseek / mock）
        usage    : token 用量，形如 {"prompt_tokens":..,"completion_tokens":..}，无则空
    """

    text: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """
    组装对话消息。

    输入：用户问题、已编号的上下文文本；输出：OpenAI 格式的 messages 列表。
    说明：上下文放在用户消息里而非系统消息，便于后续换成多轮对话时按轮拼接。
    """
    user_content = f"【参考资料】\n{context}\n\n【问题】\n{question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _first_sentence(text: str, limit: int = 80) -> str:
    """
    截取文本首句。

    输入：文本、长度上限；输出：首个句子（超长则截断加省略号）。
    """
    compact = re.sub(r"\s+", " ", text.strip())
    match = re.search(r"^(.{0,%d}?[。！？；;.!?])" % limit, compact)
    snippet = match.group(1) if match else compact[:limit]
    return snippet + ("…" if len(compact) > len(snippet) else "")


def _snippet_lines(used: list[RetrievedChunk]) -> list[str]:
    """
    把入选片段列成「N. 首句 [N]」形式的引用行。

    输入：实际入选的片段；输出：文本行列表。
    说明：编号与 format_context 的编号一致，保证和前端来源卡片严格对应。
    """
    return [f"{index}. {_first_sentence(hit.text)} [{index}]" for index, hit in enumerate(used, start=1)]


def _mock_answer(question: str, used: list[RetrievedChunk]) -> str:
    """
    生成离线占位回答。

    输入：问题、实际入选的片段；输出：模板化回答文本（含 [1][2] 标注）。
    说明：内容取自真实检索结果，所以引用链路和排版都是真的，只有「表述」是模板；
          明确标注「离线模式」避免被误当成模型输出。
    """
    lines = [f"（离线 mock 模式，未调用大模型）针对「{question}」，检索到以下资料：", ""]
    lines.extend(_snippet_lines(used))
    lines.append("")
    lines.append(f"以上依据来自：{len(used)} 条片段。配置 LLM_API_KEY 后可用真实模型生成归纳性回答。")
    return "\n".join(lines)


# 未配置密钥时的兜底横幅。措辞刻意与 mock 区分：mock 是「主动选择离线」，
# 这里是「想调模型但没配密钥」，两种状态的处置建议不同，不能混为一谈。
RETRIEVAL_ONLY_BANNER = "（未配置对话模型密钥，以下为检索到的相关原文，未经模型归纳）"


def _retrieval_only_answer(question: str, used: list[RetrievedChunk]) -> str:
    """
    生成「仅检索」兜底回答。

    输入：问题、实际入选的片段；输出：模板化回答文本（含 [1][2] 标注）。
    说明：LLM_API_KEY 未配置时替代模型回答 —— 不报错，直接把检索片段作为答案返回，
          使「入库 + 检索」这条链路在没有大模型的情况下依然可演示、可验证。
    """
    lines = [f"{RETRIEVAL_ONLY_BANNER}针对「{question}」，检索到以下资料：", ""]
    lines.extend(_snippet_lines(used))
    lines.append("")
    lines.append(f"以上依据来自：{len(used)} 条片段。配置 LLM_API_KEY 后可由模型归纳成完整回答。")
    return "\n".join(lines)


class LLMClient:
    """
    对话模型客户端。

    输入（构造参数，均可省略，省略则读 config）：
        provider    : "deepseek"（任何 OpenAI 兼容服务）或 "mock"
        api_key     : 密钥
        base_url    : 兼容端点
        model       : 模型名
        temperature : 采样温度，问答场景建议低温
        max_tokens  : 单次回答上限
    输出：通过 answer / answer_stream 返回 Answer 或文本增量。
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self._client = None       # 懒加载，mock 模式永不创建

    @property
    def is_mock(self) -> bool:
        """是否离线 mock 模式。输出：bool。"""
        return self.provider == "mock"

    @property
    def has_key(self) -> bool:
        """是否已配置对话模型密钥。输入：无；输出：bool。"""
        return bool(self.api_key)

    @property
    def is_retrieval_only(self) -> bool:
        """
        是否处于「仅检索」兜底模式。

        输入：无；输出：bool。
        说明：非 mock 但缺密钥时为 True —— 此时不调模型，直接把检索片段当回答返回，
              而不是抛错让 /api/ask 只剩一条 error 事件。
        """
        return not self.is_mock and not self.has_key

    @property
    def client(self):
        """
        懒加载 openai 客户端。

        输入：无；输出：openai.OpenAI 实例；密钥缺失时立刻抛错并给出可执行建议。
        说明：若配置了 LLM_USER_AGENT，则以 default_headers 注入。某些中转站按
              客户端身份（User-Agent）放行模型，非白名单客户端会回 403 agent_not_allowed。
        """
        if self._client is None:
            from openai import OpenAI

            if not self.api_key:
                raise RuntimeError(
                    "对话模型需要 API Key，请在 .env 中配置 LLM_API_KEY；"
                    "或设置 LLM_PROVIDER=mock 使用离线模式"
                )
            headers = {"User-Agent": config.LLM_USER_AGENT} if config.LLM_USER_AGENT else None
            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, default_headers=headers
            )
        return self._client

    # -------------------------------------------------------- 请求参数
    def _params(self, messages: list[dict[str, str]]) -> dict:
        """组装请求参数。输入：消息列表；输出：kwargs 字典。"""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _extract_text(self, resp) -> str:
        """
        从响应中取出回答正文。

        输入：openai 的非流式响应；输出：正文文本（可能为空串）。
        说明：推理型模型（deepseek-v4-flash 等）先输出 reasoning_content 再输出 content，
              两部分共享 max_tokens。预算不足时 finish_reason=length 且 content 为空，
              此时不静默返回空串（会被误当成「模型答不出」），而是抛出可执行的错误提示。
        """
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text and getattr(choice, "finish_reason", None) == "length":
            raise RuntimeError(
                f"模型 {self.model} 的输出被 max_tokens={self.max_tokens} 截断，正文为空。"
                "该模型是推理型，思考过程（reasoning_content）会先吃掉 token 预算，"
                "请在 .env 中调大 LLM_MAX_TOKENS 后重试。"
            )
        return text

    # -------------------------------------------------------- 对外接口
    def answer(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        """
        生成回答（非流式）。

        输入：
            question : 用户问题
            chunks   : 检索命中的片段（建议由 Retriever.search 产出）
        输出：Answer；无片段时不调用模型，直接返回引导性回复。
        """
        context, used = format_context(chunks)
        if not used:
            return Answer(text=NO_CONTEXT_REPLY, sources=[], model=self.model, provider=self.provider)

        if self.is_mock:
            return Answer(
                text=_mock_answer(question, used),
                sources=used,
                model="mock",
                provider="mock",
            )

        if self.is_retrieval_only:
            return Answer(
                text=_retrieval_only_answer(question, used),
                sources=used,
                model="retrieval-only",
                provider="retrieval-only",
            )

        resp = self.client.chat.completions.create(**self._params(build_messages(question, context)))
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": int(resp.usage.prompt_tokens or 0),
                "completion_tokens": int(resp.usage.completion_tokens or 0),
            }
        return Answer(
            text=self._extract_text(resp),
            sources=used,
            model=self.model,
            provider=self.provider,
            usage=usage,
        )

    def answer_stream(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        """
        流式生成回答，逐段吐出文本增量。

        输入：同 answer；输出：生成器，每次 yield 一小段文本（供前端打字机效果）。
        说明：来源列表不由本方法返回——调用方在开始流式前就已持有 chunks，
              前端先渲染引用骨架、再逐步填充正文即可。
        """
        context, used = format_context(chunks)
        if not used:
            yield NO_CONTEXT_REPLY
            return

        if self.is_mock:
            yield _mock_answer(question, used)
            return

        if self.is_retrieval_only:
            yield _retrieval_only_answer(question, used)
            return

        stream = self.client.chat.completions.create(
            **self._params(build_messages(question, context)),
            stream=True,
        )
        # 推理型模型会先流式吐 reasoning_content（这里刻意丢弃，只对用户展示正文），
        # 再吐 content。跟踪是否真的产出过正文，用于识别「预算被思考吃光」的情况。
        emitted = False
        finish_reason = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            delta = choice.delta
            # 只取 content；delta.reasoning_content 是思考过程，不外泄
            if delta and delta.content:
                emitted = True
                yield delta.content
        if not emitted and finish_reason == "length":
            raise RuntimeError(
                f"模型 {self.model} 的输出被 max_tokens={self.max_tokens} 截断，正文为空。"
                "该模型是推理型，思考过程（reasoning_content）会先吃掉 token 预算，"
                "请在 .env 中调大 LLM_MAX_TOKENS 后重试。"
            )


# 进程内单例
_LLM: LLMClient | None = None


def get_llm() -> LLMClient:
    """获取全局对话客户端。输入：无；输出：LLMClient 单例。"""
    global _LLM
    if _LLM is None:
        _LLM = LLMClient()
    return _LLM


if __name__ == "__main__":
    # 自检：python -m rag.llm "承接查验包括哪些内容"
    import sys

    from rag.retriever import Retriever

    question = sys.argv[1] if len(sys.argv) > 1 else "承接查验包括哪些内容"
    hits = Retriever().search(question)
    answer = get_llm().answer(question, hits)

    print(f"provider={answer.provider} model={answer.model} 来源数={len(answer.sources)}")
    print("-" * 56)
    print(answer.text)
    print("-" * 56)
    for i, src in enumerate(answer.sources, 1):
        print(f"[{i}] {src.source}（相似度 {src.score:.2f}）")
    if answer.usage:
        print(f"token 用量：{answer.usage}")
