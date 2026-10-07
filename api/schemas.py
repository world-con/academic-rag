

from pydantic import (
    BaseModel,
    Field,
)

# ============================================================
# 요청 모델
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=1,
        description="사용자가 입력한 질문")



# ============================================================
# Chunk 결과
# ============================================================

class ChunkResult(BaseModel):
    chunk_id: str | None = None

    document_id: str | None = None

    source_file: str | None = None

    page_number: int | None = None

    chunk_number: int | None = None

    char_count: int | None = None

    retrieval_rank: int | None = None

    retrieval_score: float | None = None

    rerank_rank: int | None = None

    rerank_score: float | None = None

    text: str


# ============================================================
# RAG 응답
# ============================================================

class AskResponse(BaseModel):

    question: str

    retrieved_chunks: list[ChunkResult]

    reranked_chunks: list[ChunkResult]

    context: str | None = None

    answer: str