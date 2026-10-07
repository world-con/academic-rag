# Langchain에서 Ollama가 동작하는지 확인.
from langchain_ollama import ChatOllama

from config import (
    OLLAMA_MODEL_NAME
)


print(
    f"[INFO] Ollama Model: "
    f"{OLLAMA_MODEL_NAME}"
)


llm = ChatOllama(

    model=OLLAMA_MODEL_NAME,

    temperature=0,

    reasoning=False,

    validate_model_on_init=True
)


response = llm.invoke(
    "안녕하세요. 한 문장으로 자기소개해 주세요."
)


print()
print("========================================")
print("Ollama Response")
print("========================================")

print(
    response.content
)