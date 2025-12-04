# rag.py
"""
RAG 核心：从 Chroma 中检索与问题相关的 GBM 指南 / 指南类文献片段，
构造上下文，调用 GPT-5.1 生成回答。

提供主函数：
    answer_question(question: str, top_k: int = 8) -> (answer: str, sources: list)
"""

from __future__ import annotations

import textwrap
from typing import List, Dict, Tuple

import chromadb

from config import (
    client,
    GPT_MODEL,
    EMBED_MODEL,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
)


def embed_text(text: str) -> List[float]:
    """对单个文本调用 embedding 接口。"""
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=[text],
    )
    return resp.data[0].embedding


def get_relevant_chunks(question: str, top_k: int = 8) -> List[Dict]:
    """
    在 Chroma 中检索与问题最相关的 top_k 个 chunk。
    返回列表，每个元素包含：
      {
        "text": chunk 文本,
        "meta": 该 chunk 的元数据,
        "distance": 相似度距离
      }
    """
    # 注意：这里要把 Path 转成 str，否则会报类型错误
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)

    q_emb = embed_text(question)

    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    items = []
    for doc, meta, dist in zip(docs, metas, dists):
        items.append(
            {
                "text": doc,
                "meta": meta,
                "distance": dist,
            }
        )
    return items


def build_context(chunks: List[Dict]) -> str:
    """
    把检索到的 chunk 整理成“带来源标记的上下文字符串”。

    每段前面加 [source_i]，方便在回答里引用。
    """
    parts = []
    for i, item in enumerate(chunks):
        meta = item["meta"]
        label = f"[source_{i+1}]"

        source_type = meta.get("source_type")

        if source_type in {"pubmed", "pubmed_guideline"}:
            title = meta.get("title", "") or ""
            year = meta.get("year", "") or ""
            pmid = meta.get("pmid", "") or ""
            header = f"{label} PubMed PMID {pmid} ({year}) - {title}"
        else:
            gname = meta.get("guideline_name") or meta.get("file_name") or "Guideline"
            year = meta.get("year", "") or ""
            header = f"{label} Guideline {gname} ({year})"

        body = item["text"].strip()
        parts.append(header + "\n" + body)

    return "\n\n" + "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """
    构造完整的 prompt（放在一个字符串里给模型）。
    注意：这里把 system 指令和上下文都拼进一个大字符串，
    这样不管用 Responses API 还是 Chat Completions 都能直接塞进 user 里。
    """
    system_instructions = textwrap.dedent(
        """
        你是一个专门回答“成人胶质母细胞瘤（GBM）临床指南和指南类文献”相关问题的助手。

        - 现在给你的是已经检索好的指南 / 共识 / 指南类文章片段（可能不完整，也可能存在相互矛盾的地方）。
        - 你需要严格基于这些片段作答，不要凭空编造不存在的研究或指南。
        - 当不同来源观点不一致时，请指出差异并说明可能原因（如指南版本、发表年份、证据等级不同）。
        - 如果上下文不足以支持一个确定结论，请明确说“根据当前检索到的证据无法下确定结论”，而不是硬编。
        - 回答使用中文，但保留关键英文缩写（如 GBM, MGMT, IDH, TMZ 等）。
        - 在回答中尽量引用 [source_1] [source_2] 这样的标记，让读者知道依据来自哪里。
        - 不给个体患者直接做治疗决策，只讨论证据和指南层面的推荐。
        """
    ).strip()

    user_part = f"问题：{question}\n\n请根据下面提供的文献 / 指南片段回答：\n{context}"

    prompt = system_instructions + "\n\n" + user_part
    return prompt


def call_gpt(prompt: str) -> str:
    """
    调用 GPT 模型生成回答。

    - 如果当前 openai SDK 支持 client.responses（新版），优先使用 Responses API。
    - 如果不支持（你现在的情况），自动退回到 chat.completions 接口。
    """
    # 新版 SDK 分支
    if hasattr(client, "responses"):
        resp = client.responses.create(
            model=GPT_MODEL,
            input=prompt,
        )
        # 新版 SDK 提供方便的 output_text 属性来直接拿到合并后的纯文本
        return resp.output_text

    # 兼容：老一点版本的 openai-python，用 Chat Completions
    # 这里我们把整个 prompt 作为 user 消息发过去。
    chat = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant for GBM clinical guideline Q&A.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return chat.choices[0].message.content


def answer_question(question: str, top_k: int = 8) -> Tuple[str, List[Dict]]:
    """
    对外暴露的主函数：
    - 输入：问题字符串
    - 输出： (模型回答文本, 检索到的 chunk 列表)
    """
    chunks = get_relevant_chunks(question, top_k=top_k)
    context = build_context(chunks)
    prompt = build_prompt(question, context)
    answer = call_gpt(prompt)
    return answer, chunks


if __name__ == "__main__":
    # 简单自测
    demo_q = "根据指南，复发 GBM 推荐的系统治疗策略有哪些？"
    ans, srcs = answer_question(demo_q, top_k=8)
    print("### 模型回答 ###\n")
    print(ans)

    print("\n### 命中的来源（检索到的 chunk）###")
    for i, s in enumerate(srcs, 1):
        meta = s["meta"]
        print(
            f"[source_{i}] {meta.get('source_type')} | "
            f"{meta.get('title') or meta.get('guideline_name') or meta.get('file_name')} "
            f"| {meta.get('year')} | "
            f"extra={meta.get('pmid') or meta.get('file_name')}"
        )
