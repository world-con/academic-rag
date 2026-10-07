# Academic RAG

대학 학칙 , 학사, 내규, 전공 가이드북 등의 학사정보를 기반으로 근거 문서를 검색하고 답변을 생성하는 RAG 기반 QA 시스템입니다.

## Architecture

text
User Question
↓
KURE-v1
↓
FAISS Retrieval
↓
BGE Reranker
↓
Ollama LLM
↓
Answer

Tech Stack
AI / RAG

- LangChain
- KURE-v1
- FAISS
- BGE Reranker
- Ollama
- Qwen3:8b
  Backend
- FastAPI
- Uvicorn
  Frontend
- React
- Vite

Project Structure

academic-rag/
├── api/
├── frontend/
├── experiments/
├── data/
├── build_index.py
├── rag_pipeline.py
├── config.py
└── requirements.txt

Current Features

- 학사정보 문서 Chunk 처리
- KURE-v1 기반 Embedding
- FAISS Vector Retrieval
- Cross-Encoder Reranking
- Ollama 기반 답변 생성
- FastAPI REST API
- React UI

## Run Backend

python -m uvicorn api.main:app --reload

## Run Frontend

cd frontend
npm install
npm run dev

## Docker

Docker 환경 구성 예정
