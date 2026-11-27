"""C6, instruction sources: text a capability reads from a source below `'operator'` never reaches the instructions, a
system or user part, or a tool definition: only tool output, or a framed data part naming its source.

Each source in `INSTRUCTION_SOURCES` gets a canary and runs at `'none'` and at `'operator'`. The `'operator'` run is the
control that the case is wired, so a pass at `'none'` means the text was lowered, not that it never arrived. For
`Memory`, which is data at every level, the control only shows the case is wired.

A source that breaks the contract today is a strict xfail linked to its fix. `create_agent`'s turns into an XPASS when
#12433 is fixed. The MCP cases also need their builds to pass `authority` into the toolset once #13117 gives it one.

This is its own module rather than part of `test_capability_contracts.py`: it starts an MCP server, so CI's FastMCP 3
lane runs it, and that lane can't build every capability the contract suite does.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability, Toolset
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    Authority,
    CapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionDef,
    FunctionModel,
    StreamFunctionDef,
)
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.memory import FileMemoryStore, Memory
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.sub_agents import DynamicAgents, SubAgent, SubAgents

from ._authority import untrusted
from ._mcp_server import MCPServer

CANARY = 'CANARY-c6'


@dataclass
class _Seen:
    """Where the canary reached the model, each place with the authority it reached it at."""

    places: list[tuple[str, Authority]] = field(default_factory=list[tuple[str, Authority]])

    def record(self, messages: list[ModelMessage], info: AgentInfo) -> None:
        """Note the canary in what a model was sent in the operator's voice: instructions, system parts, tool definitions.

        Every model in the run records, a sub-agent's too. The user channel is judged from the typed history instead
        (`record_history`), where a framed data part still says which source it came from.
        """
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            if CANARY in (message.instructions or ''):
                self.places.append(('instructions', 'operator'))
            self.places.extend(
                ('system part', 'operator')
                for part in message.parts
                if isinstance(part, SystemPromptPart) and CANARY in part.content
            )
        self.places.extend(
            (f'`{tool.name}` definition', 'operator')
            for tool in info.function_tools
            if CANARY in f'{tool.name} {tool.description} {json.dumps(tool.parameters_json_schema)}'
        )

    def record_history(self, messages: Sequence[ModelMessage]) -> None:
        """Note the canary in a run's typed request parts, at the authority each part carries it with.

        Called with each request's history before the model renders it, in every run, a sub-agent's too.
        """
        for message in messages:
            if isinstance(message, ModelRequest):
                self.places.extend(_place(part) for part in message.parts if CANARY in repr(part))


def _place(part: ModelRequestPart) -> tuple[str, Authority]:
    if isinstance(part, ToolReturnPart):
        return f'`{part.tool_name}` output', 'none'
    if isinstance(part, CapabilityPart):
        return f'`{part.kind}` part', part.source.authority
    # Anything else, a user or system part included, is above data: only tool output and framed data may carry it.
    return type(part).__name__, 'operator'


def _scripted(seen: _Seen, calls: Sequence[ToolCallPart] = ()) -> FunctionModel:
    """A model that makes `calls` one per step and then answers, recording what it was sent."""
    step = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        seen.record(messages, info)
        step += 1
        return ModelResponse(parts=[calls[step - 1]] if step <= len(calls) else [TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = respond(messages, info).parts
        if isinstance(part, ToolCallPart):
            yield {
                0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)
            }
        else:
            assert isinstance(part, TextPart)
            yield part.content

    return _TypedHistoryModel(seen, respond, stream_function=stream)


class _TypedHistoryModel(FunctionModel):
    """A `FunctionModel` that hands each request's history to `seen` while its capability parts are still typed."""

    def __init__(self, seen: _Seen, function: FunctionDef, *, stream_function: StreamFunctionDef) -> None:
        super().__init__(function, stream_function=stream_function)
        self._seen = seen

    def prepare_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters | None = None
    ) -> list[ModelMessage]:
        self._seen.record_history(messages)
        return super().prepare_messages(messages, model_request_parameters)


@dataclass
class _InstructionSource:
    """One source of text a capability can put in front of the model, holding the canary."""

    build: Callable[[Path, Authority, _Seen], Sequence[AgentCapability[None]]]
    """Writes the source's files under the directory and returns the capabilities that read them."""
    calls: Callable[[Authority], Sequence[ToolCallPart]] = lambda _: ()
    """The tool calls that bring the source's text to the model at the given authority."""
    working_dir: str = '.'
    operator_reaches: Authority = 'operator'
    """The highest authority the text reaches the model at when its source is the operator's."""
    canary_outputs: frozenset[str] = frozenset()
    """The tools whose output must carry the canary at `'none'`, so each route the case covers is checked."""
    known_break: str | None = None
    """Why the source breaks the contract today, with its fix; the `'none'` case is then a strict xfail."""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _skills(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    skill = directory / 'skills' / 'guide'
    _write(skill / 'SKILL.md', f'---\ndescription: Guide help {CANARY}.\n---\n\nGuide directions {CANARY}.\n')
    _write(skill / 'references' / 'guide.md', f'Use tabs {CANARY}.')
    return [Skills('skills', read_files=True)]


def _skills_calls(authority: Authority) -> list[ToolCallPart]:
    # Below `'operator'` the catalog is left out and these tools offer it; at `'operator'` it's in the instructions.
    if authority == 'operator':
        return []
    return [
        ToolCallPart('list_skills', {}),
        ToolCallPart('load_skill', {'name': 'guide'}),
        ToolCallPart('read_skill_file', {'skill': 'guide', 'path': 'references/guide.md'}),
    ]


def _repo_context_working_dir(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / 'packages' / 'foo' / 'AGENTS.md', f'Package rules {CANARY}.')
    return [RepoContext(home_dir=str(directory), nested_traversal=False), FileSystem()]


def _repo_context_ancestor(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / 'AGENTS.md', f'Repo rules {CANARY}.')
    (directory / 'packages' / 'foo').mkdir(parents=True)
    return [RepoContext(home_dir=str(directory), nested_traversal=False)]


def _repo_context_nested(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / 'sub' / 'AGENTS.md', f'Nested rules {CANARY}.')
    _write(directory / 'sub' / 'x.py', 'x = 1')
    return [RepoContext(home_dir=str(directory), autoload_instructions=False), FileSystem()]


def _memory(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    _write(directory / '.memory' / 'main' / 'MEMORY.md', f'- a fact {CANARY}')
    return [Memory(store=FileMemoryStore('.memory'))]


def _mcp_instructions(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    # `MCPToolset` has no authority to set yet, so both runs build the same operator-level toolset. The #13117 fix
    # has to pass `authority` in here too before this case can pass at `'none'`.
    server = MCPServer('docs', instructions=f'Server rules {CANARY}.')

    @server.tool()
    def search() -> str:
        return 'found'  # pragma: no cover

    return [Toolset(MCPToolset(server, include_instructions=True))]


def _mcp_tool_description(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    # As in `_mcp_instructions`: the #13117 fix has to pass `authority` into the toolset here.
    server = MCPServer('docs')

    @server.tool(description=f'Search the docs {CANARY}.')
    def search() -> str:
        return 'found'  # pragma: no cover

    return [Toolset(MCPToolset(server))]


def _sub_agents(directory: Path, authority: Authority, seen: _Seen) -> list[AgentCapability[None]]:
    child = _scripted(seen)
    if authority == 'operator':
        # The operator's own sub-agent: its instructions are the operator's, the control for the model-written ones.
        helper = Agent(child, name='helper', instructions=f'Helper rules {CANARY}.')
        return [SubAgents(agents=[SubAgent(helper)], agent_folders=None)]
    return [SubAgents(models={'child': child}, dynamic_agents=DynamicAgents(models=['child']), agent_folders=None)]


def _sub_agents_calls(authority: Authority) -> list[ToolCallPart]:
    delegate = ToolCallPart('delegate_task', {'agent_name': 'helper', 'task': 'Help.'})
    if authority == 'operator':
        return [delegate]
    create = {'name': 'helper', 'description': 'Helps.', 'instructions': f'Helper rules {CANARY}.', 'model': 'child'}
    return [ToolCallPart('create_agent', create), delegate]


INSTRUCTION_SOURCES: dict[str, _InstructionSource] = {
    'skills': _InstructionSource(
        _skills, _skills_calls, canary_outputs=frozenset({'list_skills', 'load_skill', 'read_skill_file'})
    ),
    'repo_context_working_dir': _InstructionSource(
        _repo_context_working_dir,
        lambda _: [ToolCallPart('read_file', {'path': 'AGENTS.md'})],
        'packages/foo',
        canary_outputs=frozenset({'read_file'}),
    ),
    'repo_context_ancestor': _InstructionSource(_repo_context_ancestor, working_dir='packages/foo'),
    'repo_context_nested': _InstructionSource(
        _repo_context_nested, lambda _: [ToolCallPart('read_file', {'path': 'sub/x.py'})]
    ),
    # Memory is data even from the operator's own store, attributed to the agent that wrote it.
    'memory': _InstructionSource(_memory, operator_reaches='none'),
    'mcp_instructions': _InstructionSource(
        _mcp_instructions,
        known_break=(
            'MCP servers have no authority yet; their text goes out as the operator (#13117). '
            'Removing this needs the fix and `authority` passed into the toolset in `_mcp_instructions`'
        ),
    ),
    'mcp_tool_description': _InstructionSource(
        _mcp_tool_description,
        known_break=(
            'MCP servers have no authority yet; their text goes out as the operator (#13117). '
            'Removing this needs the fix and `authority` passed into the toolset in `_mcp_tool_description`'
        ),
    ),
    'sub_agents_create_agent': _InstructionSource(
        _sub_agents,
        _sub_agents_calls,
        known_break='model-written instructions become the child agent system prompt (#12433)',
    ),
}


async def _reached(name: str, directory: Path, authority: Authority) -> list[tuple[str, Authority]]:
    source = INSTRUCTION_SOURCES[name]
    seen = _Seen()
    capabilities = source.build(directory, authority, seen)
    working_dir = directory / source.working_dir
    workspace = LocalWorkspaceBackend(working_dir) if authority == 'operator' else untrusted(working_dir)
    await Agent(_scripted(seen, source.calls(authority)), deps_type=type(None), capabilities=capabilities).run(
        'go', workspace=workspace
    )
    return seen.places


@pytest.mark.parametrize(
    'name',
    [
        pytest.param(
            name,
            marks=()
            if (why := source.known_break) is None
            else pytest.mark.xfail(strict=True, raises=AssertionError, reason=why),
        )
        for name, source in INSTRUCTION_SOURCES.items()
    ],
)
async def test_text_below_operator_reaches_the_model_only_as_tool_output_or_framed_data(
    name: str, tmp_path: Path
) -> None:
    places = await _reached(name, tmp_path, 'none')
    assert places, 'the canary never reached the model, so the case checks nothing'
    expected = {f'`{tool}` output' for tool in INSTRUCTION_SOURCES[name].canary_outputs}
    assert expected <= {where for where, _ in places}, places
    assert [place for place in places if place[1] != 'none'] == []


@pytest.mark.parametrize('name', list(INSTRUCTION_SOURCES))
async def test_the_same_text_from_the_operator_reaches_the_model(name: str, tmp_path: Path) -> None:
    # The control for the test above: each case is wired so its source's level is what decides where the text goes.
    # For `Memory`, data at every level, it only shows the case is wired.
    places = await _reached(name, tmp_path, 'operator')
    assert any(authority == INSTRUCTION_SOURCES[name].operator_reaches for _, authority in places), places
