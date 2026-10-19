import os
from uuid import uuid4

os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = (
    "600"  # set before importing deepeval
)
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "1800"

from deepeval.evaluate.configs import AsyncConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics.plan_quality.plan_quality import PlanQualityMetric
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

metrics = [
    PlanQualityMetric(
        threshold=0.8,
        model=model,
        include_reason=True,
    )
]


for golden in dataset.evals_iterator(
    metrics=metrics,  # type: ignore
    async_config=AsyncConfig(max_concurrent=1),
):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            "configurable": {"thread_id": str(uuid4())},
            "callbacks": [CallbackHandler()],
        },
    )
