from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="mlx-community/Qwen3.5-9B-MLX-4bit",
    base_url="http://localhost:8002/v1",
    api_key="dummy",
    temperature=0,
)
