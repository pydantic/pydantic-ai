"""Typed tool parts under Temporal's workflow sandbox.

The sandbox re-executes this module, so workflow code holds its own copy of `LookupCallPart`, while
activity code holds the host's. Parts promoted on either side must be instances of the class that
side imported, so `isinstance` works the same with and without Temporal.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from typing import Any

import pytest
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelMessage, ModelResponse, RunContext, TextPart, ToolCallPart, ToolKind
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

with workflow.unsafe.imports_passed_through():
    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE

pytestmark = [pytest.mark.xdist_group(name='temporal-durability')]


class LookupArgs(TypedDict):
    sku: str


class LookupResult(TypedDict):
    in_stock: bool


lookup_kind = ToolKind('temporal_test', 'lookup', args=LookupArgs, content=LookupResult)


class LookupCallPart(ToolCallPart[LookupArgs], kind=lookup_kind):
    pass


class LookupReturnPart(ToolReturnPart[LookupResult], kind=lookup_kind):
    pass


_workflow_checks: list[tuple[bool, bool]] = []
"""Per model request, whether workflow code recognized the lookup call and return with `isinstance`."""


@dataclass
class RecognizeLookups(AbstractCapability[Any]):
    """Checks the history with `isinstance` from workflow code, before every model request."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        parts = [part for message in request_context.messages for part in message.parts]
        _workflow_checks.append(
            (
                any(isinstance(part, LookupCallPart) for part in parts),
                any(isinstance(part, LookupReturnPart) for part in parts),
            )
        )
        return request_context


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('check_stock', {'sku': 'A-1'}, tool_call_id='c1')])
    return ModelResponse(parts=[TextPart('A-1 is in stock.')])


def check_stock(args: LookupArgs) -> LookupResult:
    return {'in_stock': args['sku'] == 'A-1'}


_agent = Agent(
    FunctionModel(_model_fn),
    name='typed_tool_parts',
    capabilities=[RecognizeLookups(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)
_agent.tool_plain(tool_kind=lookup_kind)(check_stock)


@workflow.defn
class TypedToolPartsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> list[list[bool]]:
        _workflow_checks.clear()
        await _agent.run(prompt)
        return [list(check) for check in _workflow_checks]


async def test_workflow_code_recognizes_typed_parts_with_its_own_classes(client: Client):
    async with Worker(client, task_queue=TASK_QUEUE, workflows=[TypedToolPartsWorkflow], plugins=[AgentPlugin(_agent)]):
        checks = await client.execute_workflow(
            TypedToolPartsWorkflow.run,
            args=['Is A-1 in stock?'],
            id=f'{TypedToolPartsWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )

    # The first request has no lookup yet; the second sees both its call and its return.
    assert checks == [[False, False], [True, True]]
