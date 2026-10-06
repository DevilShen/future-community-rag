"""RAGAS 评测脚本：用大模型当裁判，量化「忠实度」与「答案正确性」。

与自建的 retrieval_eval.py / generation_eval.py 互补：
    - 自建脚本测检索命中（hit@k/MRR）和字面事实匹配——快、可解释、不花 LLM；
    - 本脚本用 RAGAS 的 LLM-as-judge 测语义层面——「答案意思对但措辞不同」也能判对，
      专治 generation_eval.py 里子串匹配抓不到的「改述」类问题。

用法（在项目 eval/ 目录下运行）：
    1. 安装依赖：  pip install ragas langchain-openai datasets
    2. 运行：      python ragas_eval.py

说明：
    - 只评测「正常题」（有 gold_sources、非负样本、非脱敏专项）；
    - 裁判 LLM 默认复用 .env 里的对话模型（DeepSeek，OpenAI 兼容接口）；
    - RAGAS API 版本变动较大（0.1 小写函数 / 0.2 大写类），本脚本按 0.2+ 的类式
      写法，若 import 报错请以你已安装版本的官方文档为准微调。

注意：裁判模型建议用便宜快的小模型；若 .env 里的 LLM 是推理型（如 deepseek-v4-flash），
      当裁判会偏慢偏贵，可单独给裁判换一个模型。
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import config
from rag import Retriever, get_llm

# RAGAS 相关依赖做惰性导入：没装时给明确提示，而不是一堆裸报错。
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import Faithfulness, AnswerCorrectness
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    _HAS_RAGAS = True
except ImportError as _e:
    _HAS_RAGAS = False
    _IMPORT_ERR = _e


def load_questions(path):
    import json
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def is_normal(q):
    """只保留能配 RAGAS 的正常题：有参考答案、非负样本、非脱敏专项。"""
    return (
        not q.get("negative")
        and q.get("q_type") != "脱敏专项"
        and bool(q.get("gold_sources"))
        and bool(q.get("reference_answer"))
    )


def build_judge_llm():
    """用 .env 里的对话模型当裁判（OpenAI 兼容接口）。"""
    if not config.LLM_API_KEY:
        raise RuntimeError("未配置 LLM_API_KEY，无法创建裁判 LLM")
    judge = ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        temperature=0,          # 裁判要稳定，温度设 0
    )
    return LangchainLLMWrapper(judge)


def run_rag(questions, retriever, llm):
    """逐题跑真实检索 + 生成，产出 RAGAS 需要的四列。"""
    rows = []
    for q in questions:
        hits = retriever.search(q["question"])
        answer = (llm.answer(q["question"], hits).text or "")
        rows.append({
            "question": q["question"],
            "answer": answer,
            "contexts": [h.text for h in hits],   # 检索到的片段正文
            "ground_truth": q["reference_answer"],
        })
    return rows


def main():
    if not _HAS_RAGAS:
        print(f"[错误] 未安装 ragas / langchain-openai：{_IMPORT_ERR}")
        print("请先执行：pip install ragas langchain-openai datasets")
        return 1

    questions = load_questions(Path(__file__).parent / "questions.jsonl")
    normal = [q for q in questions if is_normal(q)]
    if not normal:
        print("[提示] 没有可评测的正常题")
        return 1
    print(f"评测 {len(normal)} 道正常题（跳过负样本与脱敏专项）")

    rows = run_rag(normal, Retriever(), get_llm())

    data = {
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    }
    dataset = Dataset.from_dict(data)

    judge = build_judge_llm()
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), AnswerCorrectness()],
        llm=judge,
    )

    print("\n===== RAGAS 结果 =====")
    print(result)

    # 逐题分（可选）：部分版本支持 to_pandas
    try:
        df = result.to_pandas()
        print("\n===== 逐题明细 =====")
        print(df.to_string(index=False))
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())