# config.py
"""
统一从项目根目录的 config.json 读取配置：
- OpenAI / GPT-5.1 / Embedding 模型
- PubMed 的 email 和 API key
- 数据和向量库路径、chunk 参数等

config.json 模板见项目根目录的 config.json 示例。
"""

import json
import os
from pathlib import Path

from openai import OpenAI

# ===== 读取 config.json =====

BASE_DIR = Path(__file__).resolve().parent

# 默认在项目根目录找 config.json，你也可以用环境变量覆盖路径
CONFIG_PATH = Path(os.getenv("GBM_RAG_CONFIG", BASE_DIR / "config.json"))

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"配置文件未找到：{CONFIG_PATH}\n"
        "请在项目根目录创建 config.json，或设置环境变量 GBM_RAG_CONFIG 指向你的配置文件。"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _CFG = json.load(f)


# ===== OpenAI / GPT 配置 =====

_OPENAI_CFG = _CFG.get("openai", {})

OPENAI_API_KEY = _OPENAI_CFG.get("api_key")
OPENAI_BASE_URL = _OPENAI_CFG.get("base_url") or None
GPT_MODEL = _OPENAI_CFG.get("model", "gpt-5.1")
EMBED_MODEL = _OPENAI_CFG.get("embed_model", "text-embedding-3-large")

if not OPENAI_API_KEY:
    raise ValueError("config.json 中的 openai.api_key 不能为空！")

_client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    _client_kwargs["base_url"] = OPENAI_BASE_URL

client = OpenAI(**_client_kwargs)


# ===== PubMed 配置 =====

_PUBMED_CFG = _CFG.get("pubmed", {})

PUBMED_EMAIL = _PUBMED_CFG.get("email")
PUBMED_API_KEY = _PUBMED_CFG.get("api_key") or None

if not PUBMED_EMAIL:
    raise ValueError("config.json 中的 pubmed.email 不能为空！")


# ===== RAG & 路径配置 =====

_RAG_CFG = _CFG.get("rag", {})
_PATH_CFG = _CFG.get("paths", {})

# chunk 设置（字符级）
CHUNK_CHAR_SIZE = int(_RAG_CFG.get("chunk_char_size", 1200))
CHUNK_CHAR_OVERLAP = int(_RAG_CFG.get("chunk_char_overlap", 200))

# 路径（相对于项目根目录）
DATA_DIR = BASE_DIR / _PATH_CFG.get("data_dir", "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CHROMA_DB_DIR = BASE_DIR / _PATH_CFG.get("chroma_db_dir", "chroma_db")
CHROMA_COLLECTION_NAME = _RAG_CFG.get("chroma_collection_name", "gbm_rag")

# 原来用到的具体文件路径
PUBMED_JSONL = RAW_DIR / "pubmed_gbm.jsonl"
GUIDELINES_JSONL = RAW_DIR / "guidelines_text.jsonl"

# 确保目录存在
for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CHROMA_DB_DIR]:
    os.makedirs(p, exist_ok=True)
