"""`DeferredToolRequests.validated_args`: whoever resolves a deferred call sees the arguments its tool will receive.

These tests use `FunctionModel`, not recordings: each case needs the model to send arguments that validation
changes in one specific way (`'false'` for a `bool`, an alias beside its field name, an unnormalized URL), which
a real model doesn't produce reliably, and nothing here depends on a provider.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, HttpUrl
from pydantic_core import to_jsonable_python
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
)
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ._inline_snapshot import snapshot


class Item(BaseModel):
    name: str


class ItemDict(TypedDict):
    name: str


@dataclass
class ItemDataclass:
    name: str


class Person(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_name: str = Field(alias='userName')


def coerce(flag: bool) -> dict[str, Any]:
    return {'flag': flag}


def model_param(item: Item) -> dict[str, Any]:
    return {'item': item}


def typed_dict_param(item: ItemDict) -> dict[str, Any]:
    return {'item': item}


def dataclass_param(item: ItemDataclass) -> dict[str, Any]:
    return {'item': item}


def aliased(person: Person) -> dict[str, Any]:
    return {'person': person}


def url(target: HttpUrl) -> dict[str, Any]:
    return {'target': target}


def transfer(amount: int) -> dict[str, Any]:
    return {'amount': amount}


class ClampAmount(AbstractCapability[Any]):
    """A validation hook that rewrites the arguments, as in https://github.com/pydantic/pydantic-ai/issues/6968."""

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        return {**args, 'amount': min(args['amount'], 50)}


@dataclass(frozen=True)
class Case:
    id: str
    tool: Callable[..., dict[str, Any]]
    model_args: dict[str, Any]
    validated_args: dict[str, Any]
    capabilities: list[AbstractCapability[Any]] = field(default_factory=list[AbstractCapability[Any]])


CASES = [
    Case(
        id='lax-coercion',
        tool=coerce,
        model_args={'flag': 'false'},
        validated_args=snapshot({'flag': False}),
    ),
    Case(
        id='base-model-drops-extra-key',
        tool=model_param,
        model_args={'item': {'name': 'a', 'extra': 'dropped'}},
        validated_args=snapshot({'item': {'name': 'a'}}),
    ),
    Case(
        id='typed-dict-drops-extra-key',
        tool=typed_dict_param,
        model_args={'item': {'name': 'a', 'extra': 'dropped'}},
        validated_args=snapshot({'item': {'name': 'a'}}),
    ),
    Case(
        id='dataclass-drops-extra-key',
        tool=dataclass_param,
        model_args={'item': {'name': 'a', 'extra': 'dropped'}},
        validated_args=snapshot({'item': {'name': 'a'}}),
    ),
    Case(
        id='alias-wins-over-field-name',
        tool=aliased,
        model_args={'person': {'user_name': 'by name', 'userName': 'by alias'}},
        validated_args=snapshot({'person': {'userName': 'by alias'}}),
    ),
    Case(
        id='url-normalized',
        tool=url,
        model_args={'target': 'HTTPS://Example.COM'},
        validated_args=snapshot({'target': 'https://example.com/'}),
    ),
    Case(
        id='after-tool-validate-rewrite',
        tool=transfer,
        model_args={'amount': 100},
        validated_args=snapshot({'amount': 50}),
        capabilities=[ClampAmount()],
    ),
]

DeferralPath = Literal['requires-approval', 'approval-required-toolset', 'approval-required-toolset-exhaustive']
"""How the call is deferred: by its tool's kind, before it runs, or by its toolset raising `ApprovalRequired`
while it runs, under the default and the `'exhaustive'` end strategy."""


@pytest.mark.parametrize(
    'path', ['requires-approval', 'approval-required-toolset', 'approval-required-toolset-exhaustive']
)
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])
async def test_approver_sees_what_the_tool_receives(case: Case, path: DeferralPath):
    """`validated_args` equals what the approved tool receives, while history keeps the model's arguments."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(case.tool.__name__, case.model_args, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    if path == 'requires-approval':
        toolset = FunctionToolset([case.tool], requires_approval=True)
    else:
        toolset = FunctionToolset([case.tool]).approval_required()
    agent = Agent(
        FunctionModel(respond),
        output_type=[str, DeferredToolRequests],
        toolsets=[toolset],
        capabilities=case.capabilities,
        end_strategy='exhaustive' if path == 'approval-required-toolset-exhaustive' else 'graceful',
    )

    result = await agent.run('go')
    requests = result.output
    assert isinstance(requests, DeferredToolRequests)
    assert requests.validated_args == {'call': case.validated_args}
    assert requests.approvals == [ToolCallPart(case.tool.__name__, case.model_args, tool_call_id='call')]

    resumed = await agent.run(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={'call': ToolApproved()}),
    )
    tool_return = next(
        part
        for message in resumed.new_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id == 'call'
    )
    assert to_jsonable_python(tool_return.content) == requests.validated_args['call']
    recorded_calls = [
        part for message in resumed.all_messages() for part in message.parts if isinstance(part, ToolCallPart)
    ]
    assert recorded_calls == [ToolCallPart(case.tool.__name__, case.model_args, tool_call_id='call')]


async def test_external_call_carries_validated_args():
    """A call deferred for external execution carries its validated arguments, and so does the run's `conversation`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('lookup', {'target': 'HTTPS://Example.COM'}, tool_call_id='call')])

    agent = Agent(FunctionModel(respond), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def lookup(target: HttpUrl) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[ToolCallPart(tool_name='lookup', args={'target': 'HTTPS://Example.COM'}, tool_call_id='call')],
            validated_args={'call': {'target': 'https://example.com/'}},
        )
    )
    assert result.conversation.deferred_tool_requests == result.output


async def test_call_deferred_again_after_inline_approval_carries_validated_args():
    """A call a handler approves that its tool then defers again joins the pending requests with its validated arguments."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('step', {'x': '1'}, tool_call_id='re1'),
                ToolCallPart('step', {'x': '2'}, tool_call_id='un1'),
            ]
        )

    async def approve_re1(ctx: RunContext[Any], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={'re1': True})

    agent = Agent(
        FunctionModel(respond),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=approve_re1)],
    )

    @agent.tool
    def step(ctx: RunContext[Any], x: int) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[ToolCallPart('step', {'x': '1'}, tool_call_id='re1')],
            approvals=[ToolCallPart('step', {'x': '2'}, tool_call_id='un1')],
            validated_args={'un1': {'x': 2}, 're1': {'x': 1}},
        )
    )


class Removal(BaseModel):
    path: str
    dry_run: bool = Field(exclude=True)


async def test_validated_args_hold_defaults_and_follow_serializer_exclusions():
    """`validated_args` holds each parameter's default, and leaves out a field its model excludes from serialization.

    The tool receives that excluded field all the same.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart('remove', {'removal': {'path': '.env', 'dry_run': True}}, tool_call_id='call')]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def remove(removal: Removal, backup_dir: str = '/srv/backups') -> bool:
        return removal.dry_run

    result = await agent.run('go')
    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.validated_args == snapshot(
        {'call': {'removal': {'path': '.env'}, 'backup_dir': '/srv/backups'}}
    )

    resumed = await agent.run(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={'call': ToolApproved()}),
    )
    tool_return = next(
        part for message in resumed.new_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    )
    assert tool_return.content is True


async def test_bytes_are_stored_as_base64():
    """A `bytes` argument is stored as base64, so one that isn't UTF-8 doesn't stop the call from being deferred.

    The model's JSON arguments only validate to UTF-8 `bytes`, so a validation hook supplies the binary value.
    """

    class LoadPayload(AbstractCapability[Any]):
        async def after_tool_validate(
            self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            return {'payload': b'\x89PNG'}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('upload', {'payload': 'image.png'}, tool_call_id='call')])

    agent = Agent(FunctionModel(respond), output_type=[str, DeferredToolRequests], capabilities=[LoadPayload()])

    @agent.tool_plain(requires_approval=True)
    def upload(payload: bytes) -> str:  # pragma: no cover
        # Never approved: the test asserts the pending request.
        return 'uploaded'

    result = await agent.run('go')
    requests = result.output
    assert isinstance(requests, DeferredToolRequests)
    assert requests.validated_args == snapshot({'call': {'payload': 'iVBORw=='}})


async def test_unserializable_value_is_stored_as_str():
    """A parameter value Pydantic can't serialize is stored as its `str()`, so the requests stay serializable."""

    class Opaque:
        def __init__(self, value: str) -> None:
            self.value = value

        def __str__(self) -> str:
            return f'Opaque({self.value})'

    class WrapOpaque(AbstractCapability[Any]):
        async def after_tool_validate(
            self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            return {'value': Opaque(args['value'])}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('use', {'value': 'x'}, tool_call_id='call')])

    agent = Agent(FunctionModel(respond), output_type=[str, DeferredToolRequests], capabilities=[WrapOpaque()])

    @agent.tool_plain(requires_approval=True)
    def use(value: str) -> str:  # pragma: no cover
        # Never approved: the test asserts the pending request.
        return value

    result = await agent.run('go')
    requests = result.output
    assert isinstance(requests, DeferredToolRequests)
    assert requests.validated_args == snapshot({'call': {'value': 'Opaque(x)'}})
