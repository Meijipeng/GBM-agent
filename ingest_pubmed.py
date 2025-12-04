# ingest_pubmed.py
"""
从 PubMed 抓取与 GBM（glioblastoma）相关、且带“免费全文（free full text）”的
“临床指南 / 共识类”文献，并保存为 JSONL 文件，用于后续 RAG。

筛选逻辑：
- 主题：glioblastoma（标题 / 摘要 / MeSH）
- 出版类型（Publication Type, pt）：
    Practice Guideline / Guideline / Consensus Development Conference
- 文本可用性：free full text[sb]

每条记录写入 config.PUBMED_JSONL，结构示例：
{
  "pmid": "25079102",
  "pmcid": "PMCxxxxxxx" 或 null,
  "doi": "10.xxxx/xxxxxxx" 或 null,
  "title": "...",
  "abstract": "...",
  "journal": "...",
  "year": "2014",
  "mesh_terms": [...],
  "pub_types": [...],
  "source_type": "pubmed_guideline"
}
"""

import json
import time
import xml.etree.ElementTree as ET

import requests
from tqdm import tqdm

from config import PUBMED_JSONL, PUBMED_EMAIL, PUBMED_API_KEY

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def build_params(extra: dict) -> dict:
    """
    在请求参数中统一附加 email / api_key，
    符合 NCBI 对 E-utilities 的使用规范。
    """
    params = {
        "tool": "gbm_rag",
        "email": PUBMED_EMAIL,
        **extra,
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    return params


def search_pubmed_ids(term: str, mindate: str, maxdate: str, retmax: int = 5000):
    """
    使用 ESearch 按检索式 + 时间范围查询 PMID 列表。
    """
    url = EUTILS_BASE + "esearch.fcgi"
    params = build_params(
        {
            "db": "pubmed",
            "term": term,
            "mindate": mindate,
            "maxdate": maxdate,
            "datetype": "pdat",
            "retmax": retmax,
            "retmode": "json",
        }
    )

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    print(f"[ingest_pubmed] ESearch got {len(pmids)} PMIDs")
    return pmids


def fetch_pubmed_xml(pmids):
    """
    使用 EFetch 批量获取 PMID 对应的 PubMed XML 详情。
    """
    if not pmids:
        return ""

    url = EUTILS_BASE + "efetch.fcgi"
    params = build_params(
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
    )

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def parse_pubmed_xml(xml_text: str):
    """
    解析 PubMed XML，提取：
    - PMID
    - PMCID（如果有）
    - DOI（如果有）
    - 标题、摘要、期刊、年份、MeSH、Publication Types
    """
    if not xml_text.strip():
        return

    root = ET.fromstring(xml_text)

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        article_info = medline.find("Article") if medline is not None else None

        pmid = medline.findtext("PMID", default="") if medline is not None else ""

        # 标题
        title = (
            article_info.findtext("ArticleTitle", default="")
            if article_info is not None
            else ""
        )

        # 摘要
        abstract_parts = []
        if article_info is not None:
            for ab in article_info.findall(".//Abstract/AbstractText"):
                text = ab.text or ""
                label = ab.get("Label")
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = "\n".join(abstract_parts).strip()

        # 期刊名
        journal = (
            article_info.findtext("Journal/Title", default="")
            if article_info is not None
            else ""
        )

        # 年份
        year = ""
        if article_info is not None:
            year = (
                article_info.findtext("Journal/JournalIssue/PubDate/Year")
                or article_info.findtext("Journal/JournalIssue/PubDate/MedlineDate")
                or ""
            )

        # MeSH
        mesh_terms = []
        if medline is not None:
            for mh in medline.findall(".//MeshHeading"):
                desc = mh.findtext("DescriptorName")
                if desc:
                    mesh_terms.append(desc)

        # Publication Types
        pub_types = []
        if article_info is not None:
            for pt in article_info.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text.strip())

        # PMCID & DOI
        pmcid = None
        doi = None
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            id_type = aid.get("IdType")
            if not id_type:
                continue
            if id_type.lower() == "pmcid" and aid.text:
                pmcid = aid.text.strip()
            elif id_type.lower() == "doi" and aid.text:
                doi = aid.text.strip()

        record = {
            "pmid": pmid,
            "pmcid": pmcid,
            "doi": doi,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "mesh_terms": mesh_terms,
            "pub_types": pub_types,
            "source_type": "pubmed_guideline",
        }

        yield record


def save_jsonl(records, path):
    count = 0
    with open(str(path), "w", encoding="utf-8") as f:
        for r in records:
            if not r.get("title") and not r.get("abstract"):
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            count += 1
    print(f"[ingest_pubmed] Saved {count} records to {path}")


def main():
    # 检索式：GBM + 指南 / 共识类 + 免费全文
    query = (
        '("glioblastoma"[Title/Abstract] OR "glioblastoma"[MeSH Terms]) '
        'AND (Practice Guideline[pt] OR Guideline[pt] '
        'OR Consensus Development Conference[pt]) '
        'AND free full text[sb]'
    )

    pmids = search_pubmed_ids(
        query,
        mindate="2010/01/01",
        maxdate="2025/12/31",
        retmax=2000,
    )

    if not pmids:
        print("[ingest_pubmed] No PMIDs found. Abort.")
        return

    batch_size = 200
    records = []

    for i in tqdm(
        range(0, len(pmids), batch_size),
        desc="[ingest_pubmed] Fetching PubMed XML",
    ):
        batch = pmids[i: i + batch_size]
        xml_text = fetch_pubmed_xml(batch)
        for rec in parse_pubmed_xml(xml_text):
            records.append(rec)
        time.sleep(0.34)

    print(f"[ingest_pubmed] Parsed {len(records)} records.")
    save_jsonl(records, PUBMED_JSONL)


if __name__ == "__main__":
    main()
