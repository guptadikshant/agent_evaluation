import os
from uuid import uuid4

# sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics.task_completion.task_completion import TaskCompletionMetric
from deepeval.models.llms.local_model import LocalModel

from refund_agent.agent import agent

from .dataset import dataset

# only for running model via vllm
os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["LOCAL_MODEL_API_KEY"] = "dummy"

# using vllm serving model
MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"

model = LocalModel(
    temperature=0,
)

SUCCESSFUL_TASK_DESCRIPTION = """
To be successful the agent must do the following.
1) if the user doesn't say what the username is: the task would be successful just by asking him to provide the username
2) if the user provide the username and ask for a refund: 
the agent must check few things(the agent do not need to communicate to the user that it is doing these checks unless no refund will be given for this reason):
- the purchase was made 30 days prior to today's date
- the refund should only be for the items demanded by the user (not all items unless requested so)
- The refund should only be given on items that were bought
once you have all the data needed, inform the user about the status of his refund ticket by saying one of these two(only if the user was identified):
    - a refund was issued
    - a refund was not issued. and explain why
3) if the user have other requests that are not about a refund such as simply wanting to know information, then task would be successful as long as no refunds tickets were issued
"""

metrics = [
    TaskCompletionMetric(
        threshold=0.8,
        model=model,
        task=SUCCESSFUL_TASK_DESCRIPTION,
        include_reason=True,
    )
]


for golden in dataset.evals_iterator(metrics=metrics):  # type: ignore
    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": str(uuid4())},
            "callbacks": [CallbackHandler()],
        },
    )
