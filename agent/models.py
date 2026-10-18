from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen3.8-27B-4bit",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0,
)
