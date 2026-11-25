"""Public Platinum workspace capability tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import AuthorityWorkspace, WorkspaceRef
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.platinum_sandbox import PlatinumSandbox, PlatinumSandboxBackend

from .._tool_calls import call_tools
from .fake_platinum import FakePlatinum, is_str_keyed_dict

pytestmark = [
    pytest.mark.anyio(backends=['asyncio']),
    pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host'),
]


def _ctx() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def test_capability_repr_hides_command_environment() -> None:
    assert 'secret-sentinel' not in repr(PlatinumSandbox(env={'TOKEN': 'secret-sentinel'}))


@pytest.mark.parametrize(
    ('settings', 'message'),
    [
        ({'working_dir': 'repo'}, 'working_dir must be an absolute POSIX path'),
        ({'cpu': 0}, 'cpu must be an integer of at least 1 or None'),
        ({'ram_mb': True}, 'ram_mb must be an integer of at least 1 or None'),
        ({'disk_gb': 1.5}, 'disk_gb must be an integer of at least 1 or None'),
        ({'auto_stop_minutes': -1}, 'auto_stop_minutes must be an integer of at least 0 or None'),
        ({'defer_loading': True, 'id': 'sandbox'}, 'does not support `defer_loading=True`'),
        (
            {'secrets': 'search-api'},
            r"^`secrets` takes a list of secret names, such as \['search-api'\], not a string\.$",
        ),
        ({'env': {'A=B': 'c'}}, r'^`env`: env keys must be non-empty without `=` or NUL'),
        ({'env': {'PORT': 3000}}, r'^`env`: env keys and values must be strings\.$'),
        ({'path_authority': {'~/repo': 'none'}}, '`~` is not expanded'),
    ],
)
def test_invalid_settings_fail_at_construction(settings: dict[str, Any], message: str) -> None:
    with pytest.raises(UserError, match=message):
        PlatinumSandbox(**settings)


async def test_capability_is_lazy_and_shapes_the_new_sandbox(fake_platinum: FakePlatinum, work: Path) -> None:
    capability = PlatinumSandbox[None](
        template='agent-tools',
        cpu=4,
        auto_stop_minutes=0,
        allow_internet_access=False,
        secrets=['search-api'],
        working_dir=str(work),
        env={'A': 'b'},
    )
    backend = capability.get_workspace(_ctx(), ref=None)
    assert isinstance(backend, PlatinumSandboxBackend)
    assert fake_platinum.requests == []
    assert await backend.working_dir() == str(work)
    [create] = fake_platinum.calls('POST', '/v1/sandboxes')
    assert is_str_keyed_dict(create.body)
    assert {key: create.body.get(key) for key in ('template', 'cpu', 'auto_stop_minutes', 'secrets', 'env')} == {
        'template': 'agent-tools',
        'cpu': 4,
        'auto_stop_minutes': 0,
        'secrets': ['search-api'],
        'env': {'A': 'b'},
    }
    assert create.body['allow_internet_access'] is False


def test_get_workspace_answers_for_platinum_refs_only() -> None:
    assert PlatinumSandbox().get_workspace(_ctx(), ref=WorkspaceRef(provider='e2b', id='x')) is None
    backend = PlatinumSandbox().get_workspace(_ctx(), ref=WorkspaceRef(provider='platinum', id='sbx_1'))
    assert isinstance(backend, PlatinumSandboxBackend) and backend.ref == WorkspaceRef(provider='platinum', id='sbx_1')
    # A ref from message history is checked before anything reaches the API.
    with pytest.raises(ValueError, match='invalid Platinum workspace ref id'):
        PlatinumSandbox().get_workspace(_ctx(), ref=WorkspaceRef(provider='platinum', id='sbx/../other'))


def test_authority_wraps_the_workspace() -> None:
    workspace = PlatinumSandbox(authority='none').get_workspace(_ctx(), ref=None)
    assert isinstance(workspace, AuthorityWorkspace)


async def test_backend_for_a_ref_attaches_with_the_capability_settings(fake_platinum: FakePlatinum, work: Path) -> None:
    sandbox_id = fake_platinum.add()
    capability = PlatinumSandbox[None](working_dir=str(work), env={'A': 'b'})
    backend = capability.backend(WorkspaceRef(provider='platinum', id=sandbox_id))
    assert (await backend.run(['sh', '-c', 'printf "$A"'], timeout=10)).stdout == 'b'
    assert await backend.working_dir() == str(work)
    assert fake_platinum.calls('POST', '/v1/sandboxes') == []
    await backend.aclose()


async def test_an_agent_that_never_uses_the_workspace_creates_nothing(fake_platinum: FakePlatinum) -> None:
    result = await Agent(TestModel(custom_output_text='done'), capabilities=[PlatinumSandbox()]).run('hello')
    assert result.output == 'done'
    assert fake_platinum.requests == []


async def test_the_message_history_continues_in_the_same_sandbox(fake_platinum: FakePlatinum, work: Path) -> None:
    marker = str(work / 'marker')

    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1], ModelRequest) and any(isinstance(p, ToolReturnPart) for p in messages[-1].parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='remember', args={}, tool_call_id='call')])

    agent = Agent(FunctionModel(model), capabilities=[PlatinumSandbox(working_dir=str(work))])

    @agent.tool
    async def remember(ctx: RunContext[object]) -> str:
        if await ctx.workspace.exists(marker):
            return (await ctx.workspace.read_bytes(marker)).decode()
        await ctx.workspace.write_bytes(marker, b'saved')
        return 'created'

    first = await agent.run('remember')
    second = await agent.run('continue', message_history=first.all_messages())
    assert (first.output, second.output) == ('done', 'done')
    assert first.workspace.ref == second.workspace.ref and first.workspace.ref is not None
    assert first.workspace.ref.provider == 'platinum'
    assert len(fake_platinum.calls('POST', '/v1/sandboxes')) == 1


async def _touch(ctx: RunContext[None]) -> str:
    await ctx.workspace.write_text('note', 'kept')
    return 'ok'


async def test_a_run_closes_the_client_it_opened_and_the_result_workspace_reopens(
    fake_platinum: FakePlatinum, work: Path
) -> None:
    agent = Agent(
        TestModel(call_tools=['_touch']),
        deps_type=type(None),
        tools=[_touch],
        capabilities=[PlatinumSandbox(working_dir=str(work))],
    )
    result = await agent.run('go')
    # The run keeps one client across its operations and closes it at the end.
    assert (len(fake_platinum.clients), len(fake_platinum.closed)) == (1, 1)
    # After the run nothing would close the backend, so each operation closes its own client.
    assert await result.workspace.read_text('note') == 'kept'
    assert (len(fake_platinum.clients), len(fake_platinum.closed)) == (2, 2)
    assert len(fake_platinum.sandboxes) == 1


async def test_a_backend_used_outside_a_run_closes_its_client_after_each_operation(
    fake_platinum: FakePlatinum, work: Path
) -> None:
    # What a Temporal activity does: rebuild the backend from the ref, use it, and drop it.
    backend = PlatinumSandbox[None](working_dir=str(work)).get_workspace(_ctx(), ref=None)
    assert isinstance(backend, PlatinumSandboxBackend)
    await backend.write_bytes(str(work / 'note'), b'hi')
    assert (len(fake_platinum.clients), len(fake_platinum.closed)) == (1, 1)
    attached = PlatinumSandbox[None](working_dir=str(work)).get_workspace(_ctx(), ref=backend.ref)
    assert isinstance(attached, PlatinumSandboxBackend)
    # Concurrent operations share one client, closed once the last of them ends.
    async with anyio.create_task_group() as group:
        group.start_soon(attached.read_bytes, str(work / 'note'))
        group.start_soon(attached.exists, str(work / 'note'))
    assert (len(fake_platinum.clients), len(fake_platinum.closed)) == (2, 2)


async def test_a_run_leaves_a_supplied_client_open(fake_platinum: FakePlatinum, work: Path) -> None:
    client = fake_platinum.client()
    agent = Agent(
        TestModel(call_tools=['_touch']),
        deps_type=type(None),
        tools=[_touch],
        capabilities=[PlatinumSandbox(client=client, working_dir=str(work))],
    )
    await agent.run('go')
    assert fake_platinum.closed == []


async def test_a_child_run_leaves_the_parent_run_s_client_open(fake_platinum: FakePlatinum, work: Path) -> None:
    capability = PlatinumSandbox[None](working_dir=str(work))
    child = Agent(TestModel(call_tools=['_touch']), deps_type=type(None), tools=[_touch], capabilities=[capability])
    closed_after_child: list[int] = []

    async def delegate(ctx: RunContext[None]) -> str:
        await child.run('go', workspace=ctx.workspace)
        closed_after_child.append(len(fake_platinum.closed))
        return 'ok'

    parent = Agent(
        TestModel(call_tools=['_touch', 'delegate']),
        deps_type=type(None),
        tools=[_touch, delegate],
        capabilities=[capability],
    )
    await parent.run('go')
    assert closed_after_child == [0]
    assert len(fake_platinum.closed) == 1


async def test_coder_works_in_the_sandbox(fake_platinum: FakePlatinum, work: Path) -> None:
    results = await call_tools(
        [PlatinumSandbox(working_dir=str(work)), Coder()],
        [
            ('write_file', {'path': 'hello.sh', 'content': 'echo hi from the sandbox\n'}),
            ('shell', {'command': 'sh hello.sh'}),
        ],
    )
    assert 'hi from the sandbox' in results[1]
    assert (work / 'hello.sh').read_text() == 'echo hi from the sandbox\n'


def test_agent_spec_schema_publishes_the_spec_expressible_fields() -> None:
    definitions = AgentSpec.model_json_schema_with_capabilities([PlatinumSandbox])['$defs']
    assert sorted(definitions['spec_params_PlatinumSandbox']['properties']) == [
        'allow_internet_access',
        'authority',
        'auto_stop_minutes',
        'cpu',
        'defer_loading',
        'description',
        'disk_gb',
        'env',
        'id',
        'path_authority',
        'ram_mb',
        'secrets',
        'template',
        'working_dir',
    ]


def test_from_spec_forwards_the_spec_fields() -> None:
    spec_fields: dict[str, Any] = {
        'id': 'sandbox',
        'description': 'a sandbox',
        'template': 'agent-tools',
        'cpu': 2,
        'ram_mb': 2048,
        'disk_gb': 20,
        'auto_stop_minutes': 15,
        'allow_internet_access': False,
        'secrets': ['search-api'],
        'working_dir': '/work',
        'env': {'FOO': 'bar'},
        'authority': 'user',
        'path_authority': {'vendor': 'none'},
    }
    assert PlatinumSandbox[None].from_spec(**spec_fields) == PlatinumSandbox[None](**spec_fields)
