# ingest_guidelines.py
import json
import os

import fitz  # PyMuPDF
from config import RAW_DIR, GUIDELINES_JSONL

GUIDELINES_DIR = os.path.join(RAW_DIR, "guidelines")
os.makedirs(GUIDELINES_DIR, exist_ok=True)


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)


def ingest_guideline_file(pdf_path: str, guideline_name: str, year: str | None = None):
    text = extract_pdf_text(pdf_path)
    record = {
        "guideline_name": guideline_name,
        "year": year,
        "text": text,
        "source_type": "guideline",
        "file_name": os.path.basename(pdf_path),
    }
    return record


def discover_guidelines():
    """
    你可以手动在这里登记文件名 -> 指南名 & 年份。
    或者用更复杂的规则自动识别，这里先简单写死示例。
    """
    mapping = []
    for fname in os.listdir(GUIDELINES_DIR):
        if not fname.lower().endswith(".pdf"):
            continue

        lower = fname.lower()
        if "nccn" in lower and "cns" in lower:
            mapping.append(
                (fname, "NCCN Guidelines: Central Nervous System Cancers", "2024")
            )
        elif "eano" in lower and "glioma" in lower:
            mapping.append(
                (fname, "EANO guideline for diffuse/malignant glioma", None)
            )
        elif "esmo" in lower and "glioma" in lower:
            mapping.append(
                (fname, "ESMO Clinical Practice Guideline for high-grade glioma", None)
            )
        else:
            # 其它指南你可以后续再按需补充
            mapping.append((fname, fname, None))

    return mapping


def main():
    records = []
    for fname, gname, year in discover_guidelines():
        pdf_path = os.path.join(GUIDELINES_DIR, fname)
        print(f"Processing guideline: {fname} -> {gname}")
        records.append(ingest_guideline_file(pdf_path, gname, year))

    with open(GUIDELINES_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved guideline texts to {GUIDELINES_JSONL}")


if __name__ == "__main__":
    main()
