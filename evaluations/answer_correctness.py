import os
from uuid import uuid4

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = (
    "600"  # set before importing deepeval
)
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "1800"

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import GEval
from deepeval.models.llms.local_model import LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

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


for golden in dataset.goldens:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": str(uuid4())},
            "callbacks": [CallbackHandler()],
        },
    )

    actual_output = response["structured_response"].answer
    tools_called = []

    for message in response["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(name=tool_call["name"], input_parameters=tool_call["args"])
            )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
    )

    dataset.add_test_case(test_case)

metrics = [
    GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Make sure that the actual output and expected output have the same logical conclusio"
        ],
        threshold=0.8,
        model=model,
        verbose_mode=True,
    ),
]

evaluate(
    test_cases=dataset.test_cases,
    metrics=metrics,
    async_config=AsyncConfig(max_concurrent=1),
)
