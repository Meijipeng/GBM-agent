# build_index.py
"""
从 PubMed 指南类记录 +（可选）指南全文 JSONL 中读取文本，
切成 chunk，调用 OpenAI embedding 生成向量，
写入本地 Chroma 向量库，用于后续 RAG 检索。

现在的策略是：

- 对 PubMed 记录：
    - 优先使用 rec["clean_text"]（也就是刚才从 PMC 抽到的全文正文）
    - 如果没有 clean_text/fulltext，就退回到 标题 + 摘要
- 对本地指南 PDF（guidelines_text.jsonl）：
    - 仍然使用其中的 text 字段
"""

import json
import os
import uuid
from typing import List, Dict, Any

import chromadb
from tqdm import tqdm

from config import (
    client,
    EMBED_MODEL,
    PUBMED_JSONL,
    GUIDELINES_JSONL,
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_CHAR_SIZE,
    CHUNK_CHAR_OVERLAP,
)


# ========== 工具函数 ==========

def load_jsonl(path) -> List[dict]:
    """读取 JSONL 文件为列表，如果文件不存在则返回空列表。"""
    if not os.path.exists(path):
        print(f"[build_index] 文件不存在，跳过：{path}")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[build_index] 从 {path} 读取到 {len(records)} 条记录")
    return records


def simple_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    简单基于字符长度的切块。
    对中英文都适用，后续你可以换成按 token 切分的方案。
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap  # 保持一定重叠

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    调用 OpenAI embedding 接口。
    一次性对一批文本做 embedding。
    """
    if not texts:
        return []

    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # 新版 SDK: resp.data 是一个对象列表，每个有 .embedding
    return [item.embedding for item in resp.data]


def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    清洗 metadata，确保所有值都是 Chroma 接受的基础类型：
    - 只允许：str / int / float / bool
    - None -> ""（空字符串）
    - list -> 用 "; " 连接成一个字符串
    - 其他类型 -> str(value)
    """
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list):
            cleaned[k] = "; ".join(str(x) for x in v)
        else:
            cleaned[k] = str(v)
    return cleaned


# ========== 主流程 ==========

def build_chroma_collection(
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
):
    """
    把所有 chunk 写入 Chroma 向量库。
    """
    if not documents:
        print("[build_index] 没有任何文档需要写入向量库，直接返回。")
        return

    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    # 注意：这里要把 Path 转成 str，否则会报类型错误
    client_chroma = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    collection = client_chroma.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[build_index] 总共 {len(documents)} 个 chunks，开始写入 Chroma...")

    batch_size = 128
    for i in tqdm(range(0, len(documents), batch_size), desc="Indexing to Chroma"):
        batch_docs = documents[i: i + batch_size]
        batch_meta_raw = metadatas[i: i + batch_size]
        batch_ids = ids[i: i + batch_size]

        batch_meta = [clean_metadata(m) for m in batch_meta_raw]

        embeddings = embed_texts(batch_docs)

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
            embeddings=embeddings,
        )

    print("[build_index] 向量索引构建完成。")


def main():
    # 1) 读取 PubMed 指南记录（现在里面已经尽量包含全文）
    pubmed_records = load_jsonl(PUBMED_JSONL)

    # 2) 读取指南全文记录（如果你暂时没有，就会是空列表）
    guideline_records = load_jsonl(GUIDELINES_JSONL)

    all_docs: List[str] = []
    all_metas: List[Dict] = []
    all_ids: List[str] = []

    # ===== PubMed 记录 chunk（优先用全文）=====
    for rec in pubmed_records:
        base_meta = {
            "source_type": rec.get("source_type", "pubmed"),  # 一般是 "pubmed_guideline"
            "pmid": rec.get("pmid"),
            "pmcid": rec.get("pmcid"),
            "title": rec.get("title"),
            "journal": rec.get("journal"),
            "year": rec.get("year"),
            "pub_types": rec.get("pub_types"),
            "has_fulltext": bool(rec.get("clean_text") or rec.get("fulltext")),
        }

        # ——关键：优先用正文 clean_text / fulltext——
        text = (
            rec.get("clean_text")
            or rec.get("fulltext")
            or f"{rec.get('title', '')}\n\n{rec.get('abstract', '')}"
        )
        text = (text or "").strip()
        if not text:
            continue

        chunks = simple_chunk_text(text, CHUNK_CHAR_SIZE, CHUNK_CHAR_OVERLAP)

        for idx, ch in enumerate(chunks):
            cid = f"pubmed-{rec.get('pmid')}-{idx}"
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            all_docs.append(ch)
            all_metas.append(meta)
            all_ids.append(cid)

    # ===== 指南 PDF 全文 chunk（如果有的话）=====
    for rec in guideline_records:
        base_meta = {
            "source_type": rec.get("source_type", "guideline"),
            "guideline_name": rec.get("guideline_name"),
            "year": rec.get("year"),
            "file_name": rec.get("file_name"),
            "url": rec.get("url"),
        }
        text = rec.get("text", "")
        if not text:
            continue

        chunks = simple_chunk_text(text, CHUNK_CHAR_SIZE, CHUNK_CHAR_OVERLAP)

        for idx, ch in enumerate(chunks):
            cid = f"guideline-{rec.get('file_name', 'guideline')}-{idx}-{uuid.uuid4().hex[:8]}"
            meta = dict(base_meta)
            meta["chunk_index"] = idx
            all_docs.append(ch)
            all_metas.append(meta)
            all_ids.append(cid)

    print(f"[build_index] 准备写入向量库的 chunk 总数：{len(all_docs)}")
    build_chroma_collection(all_docs, all_metas, all_ids)


if __name__ == "__main__":
    main()
