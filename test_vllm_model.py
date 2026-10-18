from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="Qwen3.8-27B-4bit",
    base_url="http://localhost:8000/v1",
    api_key="dummy",
    temperature=0,
)


messages = [
    SystemMessage(
        content="You are a helpful AI assistant. Provide accurate and concise answers."
    ),
    HumanMessage(content="Explain the difference between RAG and fine-tuning."),
]

response = llm.invoke(messages)

print("Response:")
print(response.content)
