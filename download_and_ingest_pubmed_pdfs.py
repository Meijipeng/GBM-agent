# download_and_ingest_pubmed_pdfs.py
"""
根据 PubMed 抓取的指南类文献（pubmed_gbm.jsonl），
尝试通过 DOI / pdf_url 下载对应的 PDF，
用 PyMuPDF 提取全文文本，写入 guidelines_text.jsonl，供后续 RAG 使用。

规则：
- 如果记录中有 rec["pdf_url"]，优先用它下载。
  （你可以手动在 pubmed_gbm.jsonl 里补充这个字段）
- 否则，如果有 rec["doi"]，构造 url = f"https://doi.org/{doi}"，
  发送请求并跟随重定向；若最终 Content-Type 包含 "pdf" 则认定为 PDF。
- 不是 pdf 的情况直接跳过（比如 publisher 显示网页，但没有直接跳到 pdf）。

注意：请仅对开放获取或你有合法访问权限的文献使用。
"""

import json
import os
import re

import fitz  # PyMuPDF
import requests
from tqdm import tqdm

from config import RAW_DIR, PUBMED_JSONL, GUIDELINES_JSONL

PDF_DIR = RAW_DIR / "article_pdfs"
os.makedirs(PDF_DIR, exist_ok=True)


def load_pubmed_records(path):
    records = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[download_pdfs] loaded {len(records)} pubmed records")
    return records


def safe_filename(name: str) -> str:
    """生成安全的文件名（只保留字母数字下划线点号）"""
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z_.-]", "", name)
    return name


def guess_pdf_url(rec: dict) -> str | None:
    """
    根据记录猜测 PDF 链接：
    1. 如果有 rec["pdf_url"]，直接用；
    2. 否则如果有 doi，用 https://doi.org/{doi}。
    """
    if rec.get("pdf_url"):
        return rec["pdf_url"]

    doi = rec.get("doi")
    if doi:
        return f"https://doi.org/{doi}"

    return None


def download_pdf(url: str, dest_path: str) -> bool:
    """
    下载单个 PDF。如果最终响应的 Content-Type 包含 pdf，则认为成功。
    """
    print(f"[download_pdfs] Trying: {url}")
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
    except Exception as e:
        print(f"[download_pdfs] Request failed: {e}")
        return False

    ctype = resp.headers.get("Content-Type", "")
    if "pdf" not in ctype.lower():
        print(f"[download_pdfs] Not a PDF (Content-Type={ctype}), skip.")
        return False

    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"[download_pdfs] Saved PDF -> {dest_path}")
    return True


def extract_pdf_text(pdf_path: str) -> str:
    """
    用 PyMuPDF 提取 PDF 全文文本。
    """
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    doc.close()
    return "\n".join(texts)


def main():
    records = load_pubmed_records(PUBMED_JSONL)

    fulltext_records = []

    for rec in tqdm(records, desc="[download_pdfs] downloading & extracting"):
        pmid = rec.get("pmid")
        title = rec.get("title", "")
        year = rec.get("year")

        url = guess_pdf_url(rec)
        if not url:
            # 没有 DOI / pdf_url，就直接跳过
            continue

        pdf_filename = safe_filename(f"{pmid or 'unknown'}.pdf")
        pdf_path = PDF_DIR / pdf_filename

        # 如果本地已经有这个 PDF，就不重复下载
        if not os.path.exists(pdf_path):
            ok = download_pdf(url, str(pdf_path))
            if not ok:
                continue

        # 提取文本
        try:
            text = extract_pdf_text(str(pdf_path))
        except Exception as e:
            print(f"[download_pdfs] Failed to extract text from {pdf_path}: {e}")
            continue

        text = (text or "").strip()
        if not text:
            print(f"[download_pdfs] Empty text for {pdf_path}, skip.")
            continue

        # 组织成指南记录，写入 guidelines_text.jsonl
        fulltext_records.append(
            {
                "guideline_name": title,
                "year": year,
                "text": text,
                "source_type": "guideline_pdf",
                "file_name": pdf_filename,
                "pmid": pmid,
                "journal": rec.get("journal"),
                "doi": rec.get("doi"),
                "original_url": url,
            }
        )

    if not fulltext_records:
        print("[download_pdfs] No fulltext records collected, nothing to write.")
        return

    # 写入 GUIDELINES_JSONL（会覆盖旧文件）
    with open(str(GUIDELINES_JSONL), "w", encoding="utf-8") as f:
        for r in fulltext_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[download_pdfs] Wrote {len(fulltext_records)} fulltext guideline records -> {GUIDELINES_JSONL}")


if __name__ == "__main__":
    main()
