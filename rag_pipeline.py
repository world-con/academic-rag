from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.vectorstores.utils import (
    DistanceStrategy
)

from langchain_community.cross_encoders import (
    HuggingFaceCrossEncoder
)

from langchain_ollama import (
    ChatOllama
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.output_parsers import (
    StrOutputParser
)

from config import (
    FAISS_PATH,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    OLLAMA_MODEL_NAME,
    RETRIEVAL_K,
    RERANK_TOP_K,
    DEVICE
)


# ============================================================
# RAG Pipeline
# ============================================================

class RAGPipeline:


    # ========================================================
    # 초기화
    # ========================================================

    def __init__(self):

        print()
        print("========================================")
        print("RAG Pipeline 초기화")
        print("========================================")

        print(
            f"[INFO] Device: {DEVICE}"
        )


        # ====================================================
        # 1. KURE-v1
        # ====================================================

        print(
            "[INFO] KURE-v1 로딩..."
        )


        self.embeddings = HuggingFaceEmbeddings(

            model_name=EMBEDDING_MODEL_NAME,

            model_kwargs={
                "device": DEVICE
            },

            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": 8
            },

            query_encode_kwargs={
                "normalize_embeddings": True
            }
        )


        # ====================================================
        # 2. 저장된 FAISS 불러오기
        # ====================================================

        print(
            "[INFO] FAISS 로딩..."
        )


        if not FAISS_PATH.exists():

            raise FileNotFoundError(
                "FAISS가 존재하지 않습니다.\n"
                "먼저 build_index.py를 실행하세요."
            )


        self.vectorstore = FAISS.load_local(

            str(FAISS_PATH),

            self.embeddings,

            allow_dangerous_deserialization=True,

            distance_strategy=(
                DistanceStrategy.MAX_INNER_PRODUCT
            )
        )


        # ====================================================
        # 3. Reranker
        # ====================================================

        print(
            "[INFO] Reranker 로딩..."
        )


        self.reranker = HuggingFaceCrossEncoder(

            model_name=RERANKER_MODEL_NAME,

            model_kwargs={
                "device": DEVICE
            }
        )


        # ====================================================
        # 4. Ollama
        # ====================================================

        print(
            "[INFO] Ollama 연결..."
        )


        self.llm = ChatOllama(

            model=OLLAMA_MODEL_NAME,

            temperature=0,

            reasoning=False,

            validate_model_on_init=True
        )


        # ====================================================
        # 5. Prompt
        # ====================================================

        self.prompt = (
            ChatPromptTemplate.from_messages(

                [

                    (
                        "system",

                        """
너는 연세대학교 학사정보를 안내하는
RAG 기반 QA 시스템이다.

다음 규칙을 반드시 지켜라.

1. 제공된 Context를 근거로만 답변한다.

2. Context에 존재하지 않는 사실을
   임의로 추측하지 않는다.

3. 학점, 학기, 기간, 날짜, 횟수 등의
   숫자 정보는 Context의 내용을 정확하게 사용한다.

4. 여러 조건이 존재한다면 빠뜨리지 말고 설명한다.

5. Context만으로 답변할 수 없다면
   "제공된 자료에서 확인할 수 없습니다."
   라고 답변한다.

6. 답변은 사용자의 질문에 먼저 직접 답하고,
   필요할 때만 추가 설명을 제공한다.

7. 답변 마지막에는 사용한 근거의
   chunk_id를 표시한다.

예시:
[근거: university_rules_p003_c07]
"""
                    ),

                    (
                        "human",

                        """
[Context]

{context}


[Question]

{question}


위 Context만 근거로 질문에 답변하세요.
"""
                    )
                ]
            )
        )


        # ====================================================
        # 6. LCEL Chain
        # ====================================================

        self.answer_chain = (

            self.prompt

            | self.llm

            | StrOutputParser()
        )


        print(
            "[INFO] RAG Pipeline 준비 완료"
        )


    # ========================================================
    # 1단계 : FAISS Retrieval
    # ========================================================

    def retrieve(
        self,
        question
    ):

        """
        KURE-v1 + FAISS를 이용해
        전체 문서에서 Top-K 후보를 찾는다.
        """


        results = (
            self.vectorstore
            .similarity_search_with_score(

                query=question,

                k=RETRIEVAL_K
            )
        )


        documents = []


        for rank, (
            document,
            score
        ) in enumerate(
            results,
            start=1
        ):

            # 검색 관련 정보를 metadata에 추가

            document.metadata[
                "retrieval_rank"
            ] = rank


            document.metadata[
                "retrieval_score"
            ] = float(score)


            documents.append(
                document
            )


        return documents


    # ========================================================
    # 2단계 : Reranking
    # ========================================================

    def rerank(
        self,
        question,
        documents
    ):

        """
        FAISS가 검색한 후보를
        Cross Encoder Reranker로 다시 평가한다.
        """


        if not documents:
            return []


        # --------------------------------------------
        # query-document 쌍 생성
        # --------------------------------------------

        pairs = [

            (
                question,
                document.page_content
            )

            for document
            in documents
        ]


        # --------------------------------------------
        # Reranker 점수 계산
        # --------------------------------------------

        scores = (
            self.reranker.score(
                pairs
            )
        )


        # --------------------------------------------
        # Document에 score 저장
        # --------------------------------------------

        for document, score in zip(
            documents,
            scores
        ):

            document.metadata[
                "rerank_score"
            ] = float(score)


        # --------------------------------------------
        # 높은 점수 순으로 정렬
        # --------------------------------------------

        reranked_documents = sorted(

            documents,

            key=lambda doc:
                doc.metadata[
                    "rerank_score"
                ],

            reverse=True
        )


        # --------------------------------------------
        # Top-K만 선택
        # --------------------------------------------

        reranked_documents = (
            reranked_documents[
                :RERANK_TOP_K
            ]
        )


        # --------------------------------------------
        # Rerank 순위 저장
        # --------------------------------------------

        for rank, document in enumerate(
            reranked_documents,
            start=1
        ):

            document.metadata[
                "rerank_rank"
            ] = rank


        return reranked_documents


    # ========================================================
    # 3단계 : Context 만들기
    # ========================================================

    def format_context(
        self,
        documents
    ):

        """
        Reranking 결과를
        LLM Prompt에 넣을 문자열로 변환한다.
        """


        context_list = []


        for index, document in enumerate(
            documents,
            start=1
        ):

            metadata = (
                document.metadata
            )


            context = f"""
[문서 {index}]

chunk_id:
{metadata.get("chunk_id")}

document_id:
{metadata.get("document_id")}

source_file:
{metadata.get("source_file")}

page_number:
{metadata.get("page_number")}

내용:
{document.page_content}
"""


            context_list.append(
                context.strip()
            )


        return "\n\n".join(
            context_list
        )


    # ========================================================
    # Document → Dictionary
    # ========================================================

    def document_to_dict(
        self,
        document
    ):

        metadata = (
            document.metadata
        )


        return {

            "chunk_id":
                metadata.get(
                    "chunk_id"
                ),

            "document_id":
                metadata.get(
                    "document_id"
                ),

            "source_file":
                metadata.get(
                    "source_file"
                ),

            "page_number":
                metadata.get(
                    "page_number"
                ),

            "chunk_number":
                metadata.get(
                    "chunk_number"
                ),

            "char_count":
                metadata.get(
                    "char_count"
                ),

            "retrieval_rank":
                metadata.get(
                    "retrieval_rank"
                ),

            "retrieval_score":
                metadata.get(
                    "retrieval_score"
                ),

            "rerank_rank":
                metadata.get(
                    "rerank_rank"
                ),

            "rerank_score":
                metadata.get(
                    "rerank_score"
                ),

            "text":
                document.page_content
        }


    # ========================================================
    # 4단계 : 전체 RAG
    # ========================================================

    def answer(
        self,
        question
    ):

        # --------------------------------------------
        # 질문 검증
        # --------------------------------------------

        question = (
            question.strip()
        )


        if not question:

            raise ValueError(
                "질문이 비어 있습니다."
            )


        # --------------------------------------------
        # 1. Retrieval
        # --------------------------------------------

        retrieved_documents = (
            self.retrieve(
                question
            )
        )


        # --------------------------------------------
        # 2. Reranking
        # --------------------------------------------

        reranked_documents = (
            self.rerank(

                question,

                retrieved_documents
            )
        )


        # --------------------------------------------
        # 검색 결과 없는 경우
        # --------------------------------------------

        if not reranked_documents:

            return {

                "question":
                    question,

                "retrieved_chunks":
                    [],

                "reranked_chunks":
                    [],

                "answer":
                    "관련 문서를 찾지 못했습니다."
            }


        # --------------------------------------------
        # 3. Context
        # --------------------------------------------

        context = (
            self.format_context(
                reranked_documents
            )
        )


        # --------------------------------------------
        # 4. Ollama
        # --------------------------------------------

        answer = (
            self.answer_chain.invoke(

                {
                    "context":
                        context,

                    "question":
                        question
                }
            )
        )


        # --------------------------------------------
        # 5. 반환
        # --------------------------------------------

        return {

            "question":
                question,

            "retrieved_chunks": [

                self.document_to_dict(
                    document
                )

                for document
                in retrieved_documents
            ],

            "reranked_chunks": [

                self.document_to_dict(
                    document
                )

                for document
                in reranked_documents
            ],

            "context":
                context,

            "answer":
                answer
        }