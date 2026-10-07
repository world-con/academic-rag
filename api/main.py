from contextlib import asynccontextmanager

import logging

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.middleware.cors import (
    CORSMiddleware
)

from config import (
    ALLOWED_ORIGINS,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    OLLAMA_MODEL_NAME
)

from rag_pipeline import (
    RAGPipeline
)

from api.schemas import (
    AskRequest,
    AskResponse
)


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# FastAPI Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    # --------------------------------------------------------
    # 서버 시작
    # --------------------------------------------------------

    logger.info(
        "RAG Pipeline 로딩 시작"
    )


    rag = RAGPipeline()


    # FastAPI 애플리케이션에 저장
    app.state.rag = rag


    logger.info(
        "RAG Pipeline 로딩 완료"
    )


    # 서버 실행
    yield


    # --------------------------------------------------------
    # 서버 종료
    # --------------------------------------------------------

    logger.info(
        "FastAPI 서버 종료"
    )


# ============================================================
# FastAPI 생성
# ============================================================

app = FastAPI(

    title="Yonsei Academic RAG API",

    description=(
        "KURE-v1 + FAISS + Reranker + "
        "Ollama 기반 학사정보 QA API"
    ),

    version="1.0.0",

    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=ALLOWED_ORIGINS,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "Yonsei Academic RAG API",

        "docs":
            "/docs"
    }


# ============================================================
# Health Check
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "ok",

        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "reranker_model":
            RERANKER_MODEL_NAME,

        "llm":
            OLLAMA_MODEL_NAME
    }


# ============================================================
# RAG API
# ============================================================

@app.post(
    "/ask",
    response_model=AskResponse
)
def ask(

    request_data: AskRequest,

    request: Request
):

    try:

        # --------------------------------------------
        # 서버 시작할 때 미리 로딩한 RAG 가져오기
        # --------------------------------------------

        rag = request.app.state.rag


        # --------------------------------------------
        # 질문
        # --------------------------------------------

        question = (
            request_data
            .question
            .strip()
        )


        # --------------------------------------------
        # RAG 실행
        # --------------------------------------------

        result = rag.answer(
            question
        )


        # --------------------------------------------
        # JSON 반환
        # --------------------------------------------

        return result


    except Exception as error:

        logger.exception(
            "RAG 처리 중 오류 발생"
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "질문 처리 중 오류가 발생했습니다."
            )

        ) from error