from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from refund_agent.models import llm
from refund_agent.objects import Answer
from refund_agent.prompt import AGENT_SYSTEM_PROMPT
from refund_agent.tools import (
    calculate_right_refund,
    get_today_date,
    get_user_purchase_history,
    issue_refund_ticket,
)

load_dotenv(find_dotenv())


agent = create_agent(
    model=llm,
    system_prompt=AGENT_SYSTEM_PROMPT,
    tools=[
        calculate_right_refund,
        get_user_purchase_history,
        issue_refund_ticket,
        get_today_date,
    ],
    response_format=Answer,
    checkpointer=InMemorySaver(),
)


def run_conversations():
    config = {"configurable": {"thread_id": str(uuid4())}}
    text = "I would to request a refund"
    print(f"User: {text}")
    question = HumanMessage(content=text)
    conversation = True

    while conversation:
        response = agent.invoke({"messages": question}, config=config)
        structured_response = response["structured_response"]
        print(f"Agent: {structured_response.answer}")

        conversation = structured_response.conversation
        if conversation:
            user_input = input("User: ")
            question = HumanMessage(content=user_input)
        print("="*80)

    print("The job is done!")


# if __name__ == "__main__":
#     run_conversations()
