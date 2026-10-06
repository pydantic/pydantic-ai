"""`open_stock_agent`: CLAI's stock agent run from code, apart from the user's saved CLAI configuration.

The agent is driven by `FunctionModel` and `TestModel`: what is under test is how CLAI assembles the
agent, not what a provider returns.
"""

from collections.abc import AsyncIterator
from contextlib import nullcontext
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import DeferredToolRequests
from pydantic_ai.capabilities import Capability, LocalWorkspace, Thinking
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness.guardrails import GuardrailResult, ToolCallInfo, ToolGuardrail
from pydantic_clai2 import _app, open_stock_agent
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins.loader import PluginSettingsError


def _answered(messages: list[ModelMessage]) -> str | None:
    """The tool result the model asked for, once it is back."""
    returned = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    return str(returned[0].content) if returned else None


@pytest.mark.parametrize('sandboxed', [False, True])
async def test_tools_and_instructions_come_from_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sandboxed: bool
) -> None:
    """The given `workspace`, or the one a host capability supplies; never the working directory."""
    for name in ('checkout', 'sandbox', 'elsewhere'):
        (tmp_path / name).mkdir()
        (tmp_path / name / 'AGENTS.md').write_text(f'Rules from the {name}.')
        (tmp_path / name / 'notes.txt').write_text(f'Notes from the {name}.')
    monkeypatch.chdir(tmp_path / 'elsewhere')
    instructions: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if (answer := _answered(messages)) is not None:
            yield answer
            return
        instructions.append(info.instructions or '')
        yield {0: DeltaToolCall(name='read_file', json_args='{"path": "notes.txt"}', tool_call_id='read')}

    async with open_stock_agent(
        workspace=str(tmp_path / 'checkout'),
        model=FunctionModel(stream_function=stream),
        capabilities=[LocalWorkspace[None](tmp_path / 'sandbox', id='sandbox')] if sandboxed else [],
    ) as agent:
        result = await agent.run('Read notes.txt')
    root = 'sandbox' if sandboxed else 'checkout'
    assert f'Notes from the {root}.' in result.output
    assert f'Rules from the {root}.' in instructions[0]


@pytest.mark.parametrize(
    'coder, delegates, disk_agents',
    [
        ({}, True, True),
        ({'agent_folders': []}, True, False),
        ({'sub_agents': False}, False, False),
    ],
)
async def test_plugin_settings_override_the_stock_coder(
    tmp_path: Path, coder: dict[str, JsonValue], delegates: bool, disk_agents: bool
) -> None:
    # The stock `agent_folders` reads personal agents from the home directory, which conftest isolates.
    personal = tmp_path / 'home' / '.claude' / 'agents'
    personal.mkdir(parents=True)
    (personal / 'reviewer.md').write_text('---\nname: reviewer\ndescription: Reviews diffs\n---\nReview the diff.')
    model = TestModel(call_tools=[], custom_output_text='done')
    async with open_stock_agent(workspace=tmp_path, model=model, plugin_settings={'coder': coder}) as agent:
        await agent.run('hello')
    parameters = model.last_model_request_parameters
    assert parameters is not None
    tools = {tool.name for tool in parameters.function_tools}
    instructions = '\n'.join(part.content for part in parameters.instruction_parts or ())
    assert 'read_file' in tools
    assert 'read_clai_customization_guide' not in tools
    assert ('delegate_task' in tools) is delegates
    assert ('Reviews diffs' in instructions) is disk_agents


async def test_plugin_settings_are_validated(tmp_path: Path) -> None:
    with pytest.raises(UserError, match="configures only 'coder', 'repo_context', 'compaction', not 'mcp'"):
        async with open_stock_agent(workspace=tmp_path, plugin_settings={'mcp': {}}):
            pass  # pragma: no cover
    with pytest.raises(PluginSettingsError, match="Plugin 'coder'"):
        async with open_stock_agent(workspace=tmp_path, plugin_settings={'coder': {'sub_agents': 'yes'}}):
            pass  # pragma: no cover


async def test_saved_configuration_does_not_apply(tmp_path: Path) -> None:
    # The user's own settings database: conftest points `XDG_CONFIG_HOME` into `tmp_path`.
    store = SettingsStore()
    store.save_plugin(PluginSettings(id='coder', factory='pydantic_clai2.builtin_plugins.coder', enabled=False))
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'extra.py').write_text(
        'from pydantic_ai.capabilities import Capability\n'
        'from pydantic_clai2.plugins import Plugin\n'
        'class Extra(Plugin):\n'
        '    def get_capabilities(self):\n'
        "        return (Capability(instructions='Saved drop-in.'),)\n"
    )
    saved = sorted(store.path.parent.rglob('*'))
    model = TestModel(call_tools=[], custom_output_text='done')
    async with open_stock_agent(workspace=tmp_path, model=model) as agent:
        await agent.run('hello')
    parameters = model.last_model_request_parameters
    assert parameters is not None
    assert 'read_file' in {tool.name for tool in parameters.function_tools}
    assert 'Saved drop-in.' not in '\n'.join(part.content for part in parameters.instruction_parts or ())
    assert sorted(store.path.parent.rglob('*')) == saved


async def test_capabilities_reach_delegated_tasks(tmp_path: Path) -> None:
    tools: dict[str, set[str]] = {}

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if (answer := _answered(messages)) is not None:
            yield answer
            return
        prompt = next(part.content for part in messages[0].parts if isinstance(part, UserPromptPart))
        caller = 'parent' if prompt == 'parent' else 'child'
        tools[caller] = {tool.name for tool in info.function_tools}
        if caller == 'parent':
            yield {
                0: DeltaToolCall(
                    name='delegate_task', json_args='{"agent_name": "self", "task": "child"}', tool_call_id='delegate'
                )
            }
        else:
            yield 'child done'

    host = Capability[None]()

    @host.tool_plain
    def lookup() -> str:
        """Look something up for the host."""
        return 'found'  # pragma: no cover

    async with open_stock_agent(
        workspace=tmp_path, model=FunctionModel(stream_function=stream), capabilities=[host]
    ) as agent:
        result = await agent.run('parent')
    assert result.output == 'child done'
    assert 'lookup' in tools['parent']
    assert 'lookup' in tools['child']


async def test_model_names_resolve_through_clai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[tuple[str, ModelSettings]] = []

    async def resolve(self: object, name: str) -> Model | str:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            requests.append((name, ModelSettings(**(info.model_settings or {}))))
            yield 'done'

        return FunctionModel(stream_function=stream, model_name=name)

    monkeypatch.setattr(_app._ModelResolver, 'resolve', resolve)  # pyright: ignore[reportPrivateUsage]
    # A setting saved for the CLI's runs on this model stays out.
    SettingsStore().save_model_settings('anthropic:claude-sonnet-4-6', {'temperature': 0.5})
    async with open_stock_agent(workspace=tmp_path, model='anthropic:claude-sonnet-4-6') as agent:
        await agent.run('hello')
        await agent.run('hello', model='openai:gpt-6')
    assert requests == snapshot(
        [
            (
                'anthropic:claude-sonnet-4-6',
                {
                    'anthropic_cache': '5m',
                    'anthropic_cache_instructions': '5m',
                    'anthropic_cache_tool_definitions': '5m',
                },
            ),
            (
                'openai:gpt-6',
                {
                    'service_tier': 'default',
                    'openai_reasoning_effort': 'medium',
                    'openai_reasoning_context': 'all_turns',
                    'openai_reasoning_mode': 'standard',
                    'openai_reasoning_summary': 'detailed',
                    'openai_text_verbosity': 'low',
                },
            ),
        ]
    )


@pytest.mark.parametrize('fail', [False, True])
async def test_plugins_get_their_settings_and_close_on_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    log = tmp_path / 'lifecycle'
    plugin = tmp_path / 'lifecycle.py'
    plugin.write_text(
        'from pathlib import Path\n'
        'from pydantic import BaseModel\n'
        'from pydantic_clai2.plugins import Plugin\n'
        'class Options(BaseModel):\n'
        "    greeting: str = 'default'\n"
        "    target: str = 'default'\n"
        'class Lifecycle(Plugin[Options]):\n'
        '    async def on_session_start(self, event):\n'
        f"        Path({str(log)!r}).write_text(self.settings.greeting + ' ' + self.settings.target)\n"
        '    async def on_session_end(self, event):\n'
        f'        p = Path({str(log)!r})\n'
        "        p.write_text(p.read_text() + ':' + event.reason)\n"
    )
    stock = PluginSettings(
        id='compaction', factory='lifecycle', path=str(plugin), settings={'greeting': 'stock', 'target': 'stock'}
    )
    monkeypatch.setattr(_app, 'STOCK_PLUGINS', (stock,))
    with pytest.raises(RuntimeError, match='host failed') if fail else nullcontext():
        async with open_stock_agent(workspace=tmp_path, plugin_settings={'compaction': {'target': 'host'}}):
            assert log.read_text() == 'stock host'
            if fail:
                raise RuntimeError('host failed')
    assert log.read_text() == f'stock host:{"error" if fail else "exit"}'


async def test_approval_gated_host_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A host's own model name, thinking, and a guardrail that defers `shell` for approval, as over ACP.

    Delegation and disk agents are off; the approved command runs without the host's LLM credentials.
    """
    monkeypatch.setenv('PYDANTIC_AI_GATEWAY_API_KEY', 'gateway-secret')
    monkeypatch.setenv('HOST_VARIABLE', 'forwarded')
    requests: list[tuple[str, ModelSettings, set[str]]] = []

    async def resolve(self: object, name: str) -> Model | str:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if (answer := _answered(messages)) is not None:
                yield answer
                return
            settings = ModelSettings(**(info.model_settings or {}))
            settings['thinking'] = info.model_request_parameters.thinking or False
            requests.append((name, settings, {tool.name for tool in info.function_tools}))
            yield {0: DeltaToolCall(name='shell', json_args='{"command": "env"}', tool_call_id='env')}

        return FunctionModel(stream_function=stream, model_name=name, profile=ModelProfile(supports_thinking=True))

    def ask_first(call: ToolCallInfo) -> GuardrailResult:
        return GuardrailResult.approve() if call.name == 'shell' else GuardrailResult.allow()

    monkeypatch.setattr(_app._ModelResolver, 'resolve', resolve)  # pyright: ignore[reportPrivateUsage]
    async with open_stock_agent(
        workspace=tmp_path,
        model='gateway/anthropic:claude-opus-5-5',
        capabilities=[Thinking(effort='high'), ToolGuardrail(guard=ask_first)],
        plugin_settings={'coder': {'sub_agents': False, 'agent_folders': []}},
    ) as agent:
        paused = await agent.run('Print the environment.', output_type=[str, DeferredToolRequests])
        assert isinstance(paused.output, DeferredToolRequests)
        assert [call.tool_name for call in paused.output.approvals] == ['shell']
        result = await agent.run(
            message_history=paused.all_messages(),
            deferred_tool_results=paused.output.build_results(approve_all=True),
            output_type=[str, DeferredToolRequests],
        )
    [(name, settings, tools)] = requests
    assert name == 'gateway/anthropic:claude-opus-5-5'
    assert settings == snapshot(
        {
            'anthropic_cache': '5m',
            'anthropic_cache_instructions': '5m',
            'anthropic_cache_tool_definitions': '5m',
            'thinking': 'high',
        }
    )
    assert 'delegate_task' not in tools
    assert isinstance(result.output, str)
    assert 'HOST_VARIABLE=forwarded' in result.output
    assert 'gateway-secret' not in result.output
