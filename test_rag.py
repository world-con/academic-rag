# 전체 파이프라인 테스트
from rag_pipeline import (
    RAGPipeline
)


# ============================================================
# 1. RAG 로딩
# ============================================================

rag = RAGPipeline()


# ============================================================
# 2. 질문
# ============================================================

question = (
    "학사학위과정의 기본 수업연한은 "
    "몇 년, 몇 학기인가?"
)


# ============================================================
# 3. RAG 실행
# ============================================================

result = rag.answer(
    question
)


# ============================================================
# 4. 질문
# ============================================================

print()
print("=" * 80)
print("QUESTION")
print("=" * 80)

print(
    result["question"]
)


# ============================================================
# 5. Retrieval 결과
# ============================================================

print()
print("=" * 80)
print("FAISS RETRIEVAL TOP-20")
print("=" * 80)


for document in result[
    "retrieved_chunks"
]:

    print()

    print(
        f"Rank: "
        f"{document['retrieval_rank']}"
    )

    print(
        f"Chunk ID: "
        f"{document['chunk_id']}"
    )

    print(
        f"Score: "
        f"{document['retrieval_score']:.4f}"
    )

    print(
        f"Source: "
        f"{document['source_file']}"
    )

    print(
        f"Page: "
        f"{document['page_number']}"
    )

    print(
        document["text"][:200]
    )


# ============================================================
# 6. Reranking 결과
# ============================================================

print()
print("=" * 80)
print("RERANK TOP-5")
print("=" * 80)


for document in result[
    "reranked_chunks"
]:

    print()

    print(
        f"Rerank Rank: "
        f"{document['rerank_rank']}"
    )

    print(
        f"Chunk ID: "
        f"{document['chunk_id']}"
    )

    print(
        f"Original Retrieval Rank: "
        f"{document['retrieval_rank']}"
    )

    print(
        f"Retrieval Score: "
        f"{document['retrieval_score']:.4f}"
    )

    print(
        f"Rerank Score: "
        f"{document['rerank_score']:.4f}"
    )

    print(
        document["text"][:300]
    )


# ============================================================
# 7. 최종 답변
# ============================================================

print()
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(
    result["answer"]
)

"""

RAG Pipeline Flow
                        ┌─────────────────────┐
                        │    chunks.csv       │
                        │     1,270개         │
                        └──────────┬──────────┘
                                   │
                              build_index.py
                                   │
                                   ▼
                              KURE-v1
                                   │
                                   ▼
                           FAISS Vector DB
                                   │
                            저장 후 재사용
                                   │
───────────────────────────────────┼────────────────────────

                              사용자 질문
                                   │
                                   ▼
                              KURE-v1
                                   │
                                   ▼
                                FAISS
                                   │
                              Top 20
                                   │
                                   ▼
                    bge-reranker-v2-m3
                                   │
                               Top 5
                                   │
                                   ▼
                         Context 생성
                                   │
                                   ▼
                    LangChain Prompt
                                   │
                                   ▼
                            ChatOllama
                                   │
                                   ▼
                             qwen3:8b
                                   │
                                   ▼
                             최종 답변
"""