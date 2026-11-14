"""The token count `UsageLimits(count_tokens_before_request=True)` asks for, under Render Workflows.

Render runs each durable unit as a child task in another process, so the count crosses the task boundary as JSON
both ways, like a model request.
"""

from __future__ import annotations

import inspect
from typing import Any

from render.workflows import TaskContext, Workflows

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext


def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('ok')])


class _CountingModel(FunctionModel):
    """Records a framework key on the counted request, as `AnthropicModel` does after a stale-thinking retry."""

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        request = next(message for message in reversed(messages) if isinstance(message, ModelRequest))
        request.metadata = {'__pydantic_ai__': {'counted': True}}
        return RequestUsage(input_tokens=5)


async def test_count_tokens_runs_in_its_own_child_task() -> None:
    """The count is a child task ahead of the request, and the key it wrote on its decoded request reaches the run."""
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](_CountingModel(_reply), name='render-count', deps_type=type(None), capabilities=[runtime])
    context = RecordingTaskContext()

    @runtime.task
    async def root(ctx: TaskContext) -> list[dict[str, Any] | None]:
        del ctx
        limits = UsageLimits(input_tokens_limit=100, count_tokens_before_request=True)
        result = await agent.run('hi', usage_limits=limits)
        return [message.metadata for message in result.all_messages() if isinstance(message, ModelRequest)]

    pending = root.func(context)
    assert inspect.isawaitable(pending)
    assert await pending == [{'__pydantic_ai__': {'counted': True}}]
    assert context.task_names == ['render-count__model.count_tokens', 'render-count__model.request']
