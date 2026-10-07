import pandas as pd

from langchain_core.documents import Document

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.vectorstores.utils import (
    DistanceStrategy
)

from config import (
    DATA_PATH,
    FAISS_PATH,
    EMBEDDING_MODEL_NAME,
    DEVICE
)


# ============================================================
# 1. CSV 로딩
# ============================================================

print()
print("========================================")
print("1. chunks.csv 로딩")
print("========================================")


df = pd.read_csv(
    DATA_PATH
)


print(
    f"[INFO] 데이터 개수: {len(df)}"
)

print(
    f"[INFO] 컬럼: {df.columns.tolist()}"
)


# ============================================================
# 2. 필요한 컬럼 확인
# ============================================================

required_columns = {

    "chunk_id",

    "document_id",

    "source_file",

    "page_number",

    "chunk_number",

    "char_count",

    "text"
}


missing_columns = (
    required_columns
    - set(df.columns)
)


if missing_columns:

    raise ValueError(
        f"필요한 컬럼이 없습니다: {missing_columns}"
    )


# ============================================================
# 3. 데이터 정리
# ============================================================

# text 없는 행 제거
df = df.dropna(
    subset=["text"]
).copy()


# 공백만 있는 text 제거
df = df[
    df["text"]
    .astype(str)
    .str.strip()
    .ne("")
].copy()


# chunk_id 없는 행 제거
df = df.dropna(
    subset=["chunk_id"]
).copy()


df = df.reset_index(
    drop=True
)


# ============================================================
# 4. chunk_id 중복 검사
# ============================================================

duplicate_count = (
    df["chunk_id"]
    .duplicated()
    .sum()
)


if duplicate_count > 0:

    raise ValueError(
        f"중복 chunk_id가 {duplicate_count}개 있습니다."
    )


print(
    f"[INFO] 사용할 Chunk 수: {len(df)}"
)

print(
    "[INFO] chunk_id 중복 없음"
)


# ============================================================
# 5. LangChain Document 생성
# ============================================================

print()
print("========================================")
print("2. LangChain Document 생성")
print("========================================")


documents = []


for _, row in df.iterrows():

    # -----------------------------------
    # metadata
    # -----------------------------------

    metadata = {

        "chunk_id":
            str(row["chunk_id"]),

        "document_id":
            str(row["document_id"]),

        "source_file":
            str(row["source_file"]),

        "page_number":
            int(row["page_number"]),

        "chunk_number":
            int(row["chunk_number"]),

        "char_count":
            int(row["char_count"])
    }


    # -----------------------------------
    # Document
    # -----------------------------------

    document = Document(

        page_content=str(
            row["text"]
        ),

        metadata=metadata
    )


    documents.append(
        document
    )


print(
    f"[INFO] Document 생성 완료: "
    f"{len(documents)}개"
)


# ============================================================
# 6. 첫 번째 Document 확인
# ============================================================

print()
print("[Document 예시]")

print(
    documents[0]
)


# ============================================================
# 7. KURE-v1 로딩
# ============================================================

print()
print("========================================")
print("3. KURE-v1 로딩")
print("========================================")


print(
    f"[INFO] Device: {DEVICE}"
)


embeddings = HuggingFaceEmbeddings(

    model_name=EMBEDDING_MODEL_NAME,

    model_kwargs={
        "device": DEVICE
    },

    # 문서 embedding
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 8
    },

    # 사용자 질문 embedding
    query_encode_kwargs={
        "normalize_embeddings": True
    },

    show_progress=True
)


# ============================================================
# 8. FAISS 생성
# ============================================================

print()
print("========================================")
print("4. FAISS 생성")
print("========================================")


vectorstore = FAISS.from_documents(

    documents=documents,

    embedding=embeddings,

    distance_strategy=(
        DistanceStrategy.MAX_INNER_PRODUCT
    )
)


# ============================================================
# 9. FAISS 저장
# ============================================================

FAISS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


vectorstore.save_local(
    str(FAISS_PATH)
)


# ============================================================
# 10. 결과
# ============================================================

print()
print("========================================")
print("FAISS 생성 완료")
print("========================================")

print(
    f"Chunk 수: {len(documents)}"
)

print(
    f"Embedding Model: "
    f"{EMBEDDING_MODEL_NAME}"
)

print(
    f"Device: {DEVICE}"
)

print(
    f"저장 위치: {FAISS_PATH}"
)