# ingest_epfl_guidelines_from_jsonl.py
"""
从本地 JSONL (epfl-llm/open_guidelines.jsonl) 中读取临床指南，
筛选与脑胶质瘤 / 脑肿瘤相关的记录，将 clean_text 写入
data/raw/guidelines_text.jsonl（config.GUIDELINES_JSONL）。

假定每行 JSON 至少包含以下字段（和 HF 数据集一致）：
- id
- source
- title
- url
- raw_text
- clean_text
- overview
"""

import json
import os

from config import DATA_DIR, RAW_DIR, GUIDELINES_JSONL

# 你的文件：/home/meijipeng/RAG-test/data/open_guidelines.jsonl
JSONL_PATH = DATA_DIR / "open_guidelines.jsonl"


def is_gbm_related(title: str, text: str) -> bool:
    """
    简单关键词过滤：只要标题或正文里出现下面任一关键词，
    就认为与脑胶质瘤 / 脑肿瘤 / 中枢神经系统肿瘤相关。
    """
    title_l = (title or "").lower()
    text_l = (text or "").lower()

    keywords = [
        "glioblastoma",
        "gbm",
        "glioma",
        "anaplastic glioma",
        "brain tumour",
        "brain tumor",
        "malignant glioma",
        "central nervous system tumour",
        "central nervous system tumor",
        "cns tumour",
        "cns tumor",
    ]

    return any(k in title_l or k in text_l for k in keywords)


def main():
    if not JSONL_PATH.exists():
        print(f"[epfl_jsonl] 找不到 JSONL 文件：{JSONL_PATH}")
        print("请确认文件路径是否正确。")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    total = 0
    selected = []

    print(f"[epfl_jsonl] 读取 JSONL: {JSONL_PATH}")

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"[epfl_jsonl] 第 {total} 行解析失败: {e}")
                continue

            clean_text = row.get("clean_text") or ""
            raw_text = row.get("raw_text") or ""
            title = row.get("title") or ""
            text = clean_text or raw_text

            if not text.strip():
                continue

            if not is_gbm_related(title, text):
                continue

            selected.append(row)

    print(f"[epfl_jsonl] 总行数: {total}")
    print(f"[epfl_jsonl] 与 GBM/脑肿瘤相关的指南条目: {len(selected)}")

    if not selected:
        print("[epfl_jsonl] 没有筛到相关指南，停止。")
        return

    # 写入 JSONL，格式与 build_index.py 兼容
    out_path = GUIDELINES_JSONL
    count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for row in selected:
            text = (row.get("clean_text") or row.get("raw_text") or "").strip()
            if not text:
                continue

            rec = {
                "guideline_name": row.get("title") or row.get("id"),
                "year": None,  # 如果以后需要可以再补年份解析
                "text": text,
                "source_type": "epfl_guideline",
                "file_name": row.get("id"),
                "url": row.get("url"),
                "source_tag": row.get("source"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"[epfl_jsonl] 已写入 {count} 条记录 -> {out_path}")


if __name__ == "__main__":
    main()
