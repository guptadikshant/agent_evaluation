from pydantic import BaseModel, Field


class Answer(BaseModel):
    answer: str = Field(description="the answer that is needed for the user question")
    conversation: bool = Field(
        description="This is an indicator to know if the conversation should continue or no. If the agent has made the decision about the refund this should be False. If the conversation still need more talk then it is True",
        default=True,
    )


class PurchaseItem(BaseModel):
    name: str = Field(description="name of the item purchased")
    