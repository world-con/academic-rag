import { useState } from "react";

import { askRag } from "./api/ragApi.js";

import "./App.css";


function App() {

  // =========================================================
  // State
  // =========================================================

  const [question, setQuestion] = useState("");

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  // =========================================================
  // 질문 전송
  // =========================================================

  const handleSubmit = async (event) => {

    event.preventDefault();


    const trimmedQuestion = question.trim();


    if (!trimmedQuestion) {

      setError(
        "질문을 입력해주세요."
      );

      return;
    }


    try {

      setLoading(true);

      setError("");

      setResult(null);


      // FastAPI 호출
      const data = await askRag(
        trimmedQuestion
      );


      setResult(data);


    } catch (error) {

      console.error(error);

      setError(
        error.message
      );


    } finally {

      setLoading(false);

    }
  };


  // =========================================================
  // UI
  // =========================================================

  return (

    <div className="app">

      {/* =====================================================
          Header
      ====================================================== */}

      <header className="header">

        <h1>
          Academic RAG
        </h1>

        <p>
          연세대학교 학사정보 QA 시스템
        </p>

      </header>


      {/* =====================================================
          Main
      ====================================================== */}

      <main className="rag-layout">


        {/* ===================================================
            1. 질문 영역
        ==================================================== */}

        <section className="panel">

          <h2>
            질문
          </h2>


          <form
            onSubmit={handleSubmit}
            className="question-form"
          >

            <textarea

              value={question}

              onChange={(event) =>
                setQuestion(
                  event.target.value
                )
              }

              placeholder={
                "학사정보에 대해 질문해 주세요."
              }

              disabled={loading}

            />


            <button
              type="submit"
              disabled={loading}
            >

              {
                loading
                  ? "검색 중..."
                  : "질문하기"
              }

            </button>

          </form>


          {
            error &&
            (
              <div className="error-message">
                {error}
              </div>
            )
          }


          {
            result &&
            (
              <div className="question-result">

                <h3>
                  입력한 질문
                </h3>

                <p>
                  {result.question}
                </p>

              </div>
            )
          }

        </section>



        {/* ===================================================
            2. 검색 결과 영역
        ==================================================== */}

        <section className="panel">

          <h2>
            검색된 Chunk
          </h2>


          {
            loading
              ? (
                  <p>
                    관련 문서를 검색하고 있습니다...
                  </p>
                )

              : !result
                ? (
                    <p className="empty-text">
                      질문을 입력하면
                      관련 Chunk가 표시됩니다.
                    </p>
                  )

                : (
                    <div className="chunk-list">

                      {
                        result.reranked_chunks.map(
                          (chunk) => (

                            <article
                              key={chunk.chunk_id}
                              className="chunk-card"
                            >

                              <div className="chunk-header">

                                <strong>
                                  #{chunk.rerank_rank}
                                </strong>

                                <span>
                                  {
                                    chunk.chunk_id
                                  }
                                </span>

                              </div>


                              <div className="chunk-meta">

                                <span>
                                  문서:
                                  {" "}
                                  {
                                    chunk.document_id
                                  }
                                </span>

                                <span>
                                  Page:
                                  {" "}
                                  {
                                    chunk.page_number
                                  }
                                </span>

                              </div>


                              <div className="chunk-score">

                                Retrieval Rank:
                                {" "}
                                {
                                  chunk.retrieval_rank
                                }

                                <br />

                                Retrieval Score:
                                {" "}
                                {
                                  chunk.retrieval_score
                                    ?.toFixed(4)
                                }

                                <br />

                                Rerank Score:
                                {" "}
                                {
                                  chunk.rerank_score
                                    ?.toFixed(4)
                                }

                              </div>


                              <p className="chunk-text">

                                {chunk.text}

                              </p>

                            </article>

                          )
                        )
                      }

                    </div>
                  )
          }

        </section>



        {/* ===================================================
            3. 최종 답변 영역
        ==================================================== */}

        <section className="panel">

          <h2>
            AI 답변
          </h2>


          {
            loading
              ? (
                  <p>
                    Ollama가 답변을 생성하고 있습니다...
                  </p>
                )

              : result
                ? (
                    <div className="answer-box">

                      {result.answer}

                    </div>
                  )

                : (
                    <p className="empty-text">
                      질문에 대한 최종 답변이
                      이곳에 표시됩니다.
                    </p>
                  )
          }

        </section>


      </main>

    </div>
  );
}


export default App;