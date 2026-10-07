from pathlib import Path

import torch


# ============================================================
# 프로젝트 경로
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "chunks.csv"

FAISS_PATH = BASE_DIR / "storage" / "faiss_index"


# ============================================================
# 모델 설정
# ============================================================

# Embedding
EMBEDDING_MODEL_NAME = "nlpai-lab/KURE-v1"

# Reranker
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Ollama LLM
OLLAMA_MODEL_NAME = "qwen3:8b"


# ============================================================
# Retrieval 설정
# ============================================================

# KURE + FAISS에서 먼저 검색할 청크 수
RETRIEVAL_K = 20

# Reranker 이후 최종적으로 LLM에게 전달할 청크 수
RERANK_TOP_K = 5


# ============================================================
# FASTAPI / CORS 설정
# ============================================================

ALLOWED_ORIGINS = [

    "http://localhost:3000",
    "http://127.0.0.1:3000",

    "http://localhost:5173",
    "http://127.0.0.1:5173"
]



# ============================================================
# Device 설정
# ============================================================

def get_device():

    # NVIDIA GPU
    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon
    if hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            return "mps"

    # GPU가 없으면 CPU
    return "cpu"


DEVICE = get_device()