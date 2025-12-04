# demo_cli.py
from rag import answer_question

def main():
    print("GBM RAG Demo - 使用 GPT-5.1 + 自建索引\n")
    print("输入你的问题（回车发送，输入 quit 退出）。\n")

    while True:
        q = input("问题> ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        answer, sources = answer_question(q, top_k=8)
        print("\n=== 模型回答 ===\n")
        print(answer)

        print("\n=== 命中的来源（前几个）===\n")
        for i, s in enumerate(sources, 1):
            meta = s["meta"]
            print(f"[source_{i}] {meta.get('source_type')} | {meta.get('title') or meta.get('guideline_name')} | {meta.get('year')} | extra={meta.get('pmid') or meta.get('file_name')}")
        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
