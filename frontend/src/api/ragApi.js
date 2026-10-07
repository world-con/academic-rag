const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL;


export async function askRag(question) {

  const response = await fetch(
    `${API_BASE_URL}/ask`,
    {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        question: question,
      }),
    }
  );


  if (!response.ok) {

    const errorData = await response
      .json()
      .catch(() => null);


    throw new Error(
      errorData?.detail ||
      "질문 처리 중 오류가 발생했습니다."
    );
  }


  return response.json();
}