"""1.x deprecation warnings for `Agent.__init__` kwargs whose migration target is a capability,
plus regression guards for kwargs already removed in v2.

Still-deprecated (warn + remap):
- `prepare_output_tools=` -> `capabilities=[PrepareOutputTools(prepare_output_tools)]`
- `history_processors=` (on `Agent.from_spec` / `Agent.from_file`) -> `ProcessHistory` capability

Removed in v2 (must raise `UserError`):
- `event_stream_handler=` (ctor kwarg removed; run-level kwarg on run()/iter() unaffected)
- `prepare_tools=` (use `PrepareTools` capability)
- `instrument=` (use `capabilities=[Instrumentation(...)]`; runtime paths dropped in #5434,
  this PR drops the leftover entry from the deprecated `__init__` overload)

`output_validators=` is not an `Agent.__init__` kwarg in 1.x (it's only set via decorator),
so it's intentionally excluded.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import PrepareOutputTools, ProcessHistory
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import RunContext, ToolDefinition

pytestmark = [pytest.mark.anyio]


# --- removed kwargs (regression guards) -------------------------------------------------


def _noop(*args: object, **kwargs: object) -> None:
    """Placeholder callable for kwargs that are dropped in v2; `validate_empty_kwargs` raises
    before the value would ever be invoked."""


def test_event_stream_handler_kwarg_raises_in_v2():
    """`Agent(event_stream_handler=...)` was dropped in v2; passing it must now raise `UserError`
    via the `**_deprecated_kwargs` -> `validate_empty_kwargs` path."""
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`event_stream_handler`'):
        Agent(TestModel(), event_stream_handler=_noop)  # pyright: ignore[reportCallIssue]


def test_prepare_tools_kwarg_raises_in_v2():
    """`Agent(prepare_tools=...)` was dropped in v2; users migrate to `capabilities=[PrepareTools(...)]`."""
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`prepare_tools`'):
        Agent(TestModel(), prepare_tools=_noop)  # pyright: ignore[reportCallIssue]


def test_from_spec_event_stream_handler_kwarg_raises_in_v2():
    """`Agent.from_spec(event_stream_handler=...)` was dropped in v2; the kwarg now hits
    `from_spec`'s own `validate_empty_kwargs` path (independent of `Agent.__init__`'s)."""
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`event_stream_handler`'):
        Agent.from_spec({'model': 'test'}, event_stream_handler=_noop)  # pyright: ignore[reportCallIssue]


def test_from_spec_prepare_tools_kwarg_raises_in_v2():
    """`Agent.from_spec(prepare_tools=...)` was dropped in v2; users migrate to
    `capabilities=[PrepareTools(...)]`."""
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`prepare_tools`'):
        Agent.from_spec({'model': 'test'}, prepare_tools=_noop)  # pyright: ignore[reportCallIssue]


def test_from_file_event_stream_handler_kwarg_raises_in_v2(tmp_path: Any):
    """`Agent.from_file` has its own `**_deprecated_kwargs` + `validate_empty_kwargs` path
    (it runs validation before forwarding to `from_spec`), so the regression guard must cover
    it independently."""
    spec_file = tmp_path / 'agent.json'
    spec_file.write_text('{"model": "test"}')
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`event_stream_handler`'):
        Agent.from_file(spec_file, event_stream_handler=_noop)  # pyright: ignore[reportCallIssue]


def test_from_file_prepare_tools_kwarg_raises_in_v2(tmp_path: Any):
    """`Agent.from_file` has its own `**_deprecated_kwargs` + `validate_empty_kwargs` path
    (it runs validation before forwarding to `from_spec`), so the regression guard must cover
    it independently."""
    spec_file = tmp_path / 'agent.json'
    spec_file.write_text('{"model": "test"}')
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`prepare_tools`'):
        Agent.from_file(spec_file, prepare_tools=_noop)  # pyright: ignore[reportCallIssue]


def test_instrument_kwarg_raises_in_v2():
    """`Agent(instrument=...)` was dropped in #5434 in favor of `capabilities=[Instrumentation(...)]`;
    the runtime consumer is gone, so the kwarg now hits `validate_empty_kwargs`."""
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`instrument`'):
        Agent(TestModel(), instrument=True)  # pyright: ignore[reportCallIssue]


def test_from_spec_instrument_kwarg_raises_in_v2():
    """`Agent.from_spec(instrument=...)` was dropped in #5434; the kwarg now hits `from_spec`'s
    own `validate_empty_kwargs` path."""
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`instrument`'):
        Agent.from_spec({'model': 'test'}, instrument=True)  # pyright: ignore[reportCallIssue]


def test_from_file_instrument_kwarg_raises_in_v2(tmp_path: Any):
    """`Agent.from_file(instrument=...)` was dropped in #5434; `from_file` validates its own
    `**_deprecated_kwargs` before forwarding to `from_spec`."""
    spec_file = tmp_path / 'agent.json'
    spec_file.write_text('{"model": "test"}')
    with pytest.raises(UserError, match=r'Unknown keyword arguments:.*`instrument`'):
        Agent.from_file(spec_file, instrument=True)  # pyright: ignore[reportCallIssue]


# --- prepare_output_tools= ---------------------------------------------------------------


async def _noop_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return tool_defs


async def test_prepare_output_tools_kwarg_emits_deprecation_warning():
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\(prepare_output_tools=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=_noop_prep)  # pyright: ignore[reportDeprecated]


async def test_prepare_output_tools_kwarg_warning_points_at_capability():
    with pytest.warns(
        PydanticAIDeprecationWarning, match=r'capabilities=\[PrepareOutputTools\(prepare_output_tools\)\]'
    ):
        Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=_noop_prep)  # pyright: ignore[reportDeprecated]


async def test_prepare_output_tools_kwarg_remaps_to_capability():
    """The kwarg auto-injects a `PrepareOutputTools` capability into the agent's capability list,
    and the prepare callback fires once during a run."""
    seen: list[int] = []

    async def prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        seen.append(len(tool_defs))
        return tool_defs

    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_output_tools'):
        agent = Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=prep)  # pyright: ignore[reportDeprecated]

    assert any(isinstance(cap, PrepareOutputTools) for cap in agent._root_capability.capabilities)  # pyright: ignore[reportPrivateUsage]
    await agent.run('hello')
    assert seen, 'prepare_output_tools callback should have fired'


async def test_prepare_output_tools_kwarg_vs_capability_equivalence():
    """`Agent(prepare_output_tools=fn)` and `Agent(capabilities=[PrepareOutputTools(fn)])` produce
    the same observable behavior."""
    kwarg_calls: list[int] = []
    cap_calls: list[int] = []

    async def kwarg_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        kwarg_calls.append(len(tool_defs))
        return tool_defs

    async def cap_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        cap_calls.append(len(tool_defs))
        return tool_defs

    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_output_tools'):
        kwarg_agent = Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=kwarg_prep)  # pyright: ignore[reportDeprecated]
    cap_agent = Agent(TestModel(), output_type=ToolOutput(str), capabilities=[PrepareOutputTools(cap_prep)])

    kwarg_result = await kwarg_agent.run('hello')
    cap_result = await cap_agent.run('hello')

    assert kwarg_result.output == cap_result.output
    assert kwarg_calls == cap_calls


# --- from_spec / from_file forwarders --------------------------------------------------------


async def test_from_file_history_processors_kwarg_routes_through_extra_capabilities(tmp_path: Any):
    """`Agent.from_file(history_processors=...)` warns and the legacy kwarg remaps into the
    `extra_capabilities` -> `merged_capabilities` path forwarded to `from_spec`."""

    def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages  # pragma: no cover

    spec_file = tmp_path / 'agent.json'
    spec_file.write_text('{"model": "test"}')

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`Agent\.from_file\(history_processors='):
        agent: Agent[Any, Any] = Agent.from_file(spec_file, history_processors=[processor])  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    process_history_caps = [cap for cap in agent._root_capability.capabilities if isinstance(cap, ProcessHistory)]  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownVariableType]
    assert process_history_caps, 'from_file(history_processors=) should remap into a ProcessHistory capability'


async def test_from_spec_prepare_output_tools_kwarg_routes_through_extra_capabilities():
    """`Agent.from_spec(prepare_output_tools=...)` warns and the legacy kwarg flows through the
    `extra_capabilities` -> `all_capabilities.extend(...)` branch onto the constructed agent."""
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`Agent\.from_spec\(prepare_output_tools='):
        agent: Agent[Any, Any] = Agent.from_spec(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
            {'model': 'test'}, output_type=ToolOutput(str), prepare_output_tools=_noop_prep
        )

    prep_caps = [cap for cap in agent._root_capability.capabilities if isinstance(cap, PrepareOutputTools)]  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownVariableType]
    assert prep_caps, 'from_spec(prepare_output_tools=) should remap into a PrepareOutputTools capability'
    result = await agent.run('hello')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    assert result.output  # pyright: ignore[reportUnknownMemberType]
