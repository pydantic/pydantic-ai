"""Tests for the `ManagedPrompt` capability (source package `pydantic_ai_harness.logfire`).

This directory is deliberately named `logfire_variables`, not `logfire`, even though it
tests the `pydantic_ai_harness.logfire` package. The pyright config scopes test-only report
overrides with `executionEnvironments = [{ root = 'tests' }]`, which makes `tests/` an import
root -- so a `tests/logfire/` directory would shadow the third-party `logfire` package for
every test file's `import logfire`. Keeping the directory off that name avoids the collision.

Style follows `tests/code_mode/test_code_mode.py`: module-level
`pytestmark = pytest.mark.anyio` and an `anyio_backend` fixture. All resolution runs
against the code default (no Logfire provider is configured), which is exactly the
safety-net behavior `ManagedPrompt` relies on. Each test uses a unique slug because the
default Logfire instance keeps its variable registry across `configure()` calls.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import logfire
import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness import ManagedPrompt
from pydantic_ai_harness.logfire import ManagedPrompt as ManagedPromptFromPackage

pytestmark = pytest.mark.anyio

DEFAULT = 'You are a helpful assistant.'


@pytest.fixture(autouse=True, scope='module')
def _configure_logfire() -> None:
    """Configure Logfire once so variable resolution does not warn (warnings are errors)."""
    logfire.configure(send_to_logfire=False, console=False)


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def instructions_seen(result_messages: list[ModelMessage]) -> list[str]:
    """Collect the rendered instructions from each `ModelRequest` in a run."""
    return [m.instructions for m in result_messages if isinstance(m, ModelRequest) and m.instructions is not None]


# Span attributes whose values vary between runs (random ids, line numbers, the
# resolution span's merged-into-attributes JSON blob from Logfire) and would otherwise
# make snapshots non-deterministic. `attributes` here is the literal key Logfire emits
# on the resolve span containing the serialized targeting attributes -- it shadows the
# enclosing span attributes dict by name, so the pop targets the inner one.
# `logfire.metrics` only appears on logfire versions newer than the extra's floor,
# so keeping it would make the snapshots depend on the resolved logfire version.
# `model_request_parameters` is pydantic-ai's serialized internal request state; its
# schema grows with pydantic-ai releases (e.g. 2.14.0 added `ToolDefinition.toolset_id`),
# so keeping it would break the `test-latest` job on every such release. These tests
# assert baggage propagation, not the request payload.
_VOLATILE_SPAN_ATTRIBUTES = (
    'attributes',
    'code.lineno',
    'gen_ai.conversation.id',
    'gen_ai.agent.call.id',
    'logfire.metrics',
    'model_request_parameters',
)


@contextmanager
def _variables_provider_configured(capfire: CaptureLogfire, variables_config: VariablesConfig) -> Generator[None]:
    """Reconfigure Logfire with a local variables provider for the duration of the block.

    Restores the module's baseline configuration on exit so the change does not leak
    into other tests in this module (or any module collected after it).
    """
    logfire.configure(
        send_to_logfire=False,
        console=False,
        variables=logfire.LocalVariablesOptions(config=variables_config),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    try:
        yield
    finally:
        logfire.configure(send_to_logfire=False, console=False)


def span_attributes(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """Each exported span as `{name, attributes}`, with volatile attributes dropped.

    Names identify which span the attributes belong to; everything else (ids, timing,
    parentage) is omitted to keep the snapshots focused and stable.
    """
    result: list[dict[str, Any]] = []
    for span in capfire.exporter.exported_spans_as_dict():
        attributes = span['attributes']
        for key in _VOLATILE_SPAN_ATTRIBUTES:
            attributes.pop(key, None)
        result.append({'name': span['name'], 'attributes': attributes})
    return result


def test_public_reexport() -> None:
    assert ManagedPrompt is ManagedPromptFromPackage


def test_slug_becomes_prompt_variable_name() -> None:
    capability = ManagedPrompt('support_agent', default=DEFAULT)
    assert capability._variable.name == 'prompt__support_agent'


def test_hyphenated_slug_is_normalized() -> None:
    capability = ManagedPrompt('welcome-email', default=DEFAULT)
    assert capability._variable.name == 'prompt__welcome_email'


def test_slug_requires_default() -> None:
    with pytest.raises(TypeError, match='`default` is required'):
        ManagedPrompt('no_default_slug')


def test_explicit_logfire_instance_is_used() -> None:
    capability = ManagedPrompt('with_instance', default=DEFAULT, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)
    assert capability._variable.name == 'prompt__with_instance'


def test_duplicate_slug_is_allowed() -> None:
    # Each ManagedPrompt builds its own backing variable, so the same slug can be declared
    # repeatedly (e.g. shared across agents) without the duplicate-registration error
    # `logfire.var` would raise.
    first = ManagedPrompt('shared_slug', default=DEFAULT)
    second = ManagedPrompt('shared_slug', default=DEFAULT)
    assert first._variable.name == second._variable.name == 'prompt__shared_slug'


def test_prompt_prefix_in_slug_warns_and_is_stripped() -> None:
    with pytest.warns(UserWarning, match='added automatically'):
        capability = ManagedPrompt('prompt__already_prefixed', default=DEFAULT)
    assert capability._variable.name == 'prompt__already_prefixed'


def test_invalid_slug_raises() -> None:
    with pytest.raises(ValueError, match='invalid variable name'):
        ManagedPrompt('has spaces', default=DEFAULT)


async def test_resolves_default_into_instructions() -> None:
    agent = Agent(TestModel(), capabilities=[ManagedPrompt('default_slug', default=DEFAULT)])

    result = await agent.run('hello')

    assert instructions_seen(result.all_messages()) == [DEFAULT]


async def test_accepts_prebuilt_variable() -> None:
    var = logfire.var(name='prompt__prebuilt', type=str, default=DEFAULT)
    agent = Agent(TestModel(), capabilities=[ManagedPrompt(var)])

    result = await agent.run('hello')

    assert instructions_seen(result.all_messages()) == [DEFAULT]


async def test_override_is_reflected() -> None:
    capability = ManagedPrompt('override_slug', default=DEFAULT)
    agent = Agent(TestModel(), capabilities=[capability])

    with capability._variable.override('Be terse.'):
        result = await agent.run('hello')

    assert instructions_seen(result.all_messages()) == ['Be terse.']


async def test_records_variable_resolution_span(capfire: CaptureLogfire) -> None:
    agent = Agent(TestModel(), capabilities=[ManagedPrompt('span_slug', default=DEFAULT)])

    await agent.run('hello')

    # Without `Instrumentation` the only span is the one Logfire records for resolving the
    # prompt variable -- the resolved value, label, version, and reason are captured as attributes.
    assert span_attributes(capfire) == snapshot(
        [
            {
                'name': 'Resolve variable prompt__span_slug',
                'attributes': {
                    'code.filepath': '_managed_prompt.py',
                    'code.function': 'wrap_run',
                    'targeting_key': 'null',
                    'logfire.msg_template': 'Resolve variable prompt__span_slug',
                    'logfire.msg': 'Resolve variable prompt__span_slug',
                    'logfire.span_type': 'span',
                    'name': 'prompt__span_slug',
                    'value': '"You are a helpful assistant."',
                    'label': 'null',
                    'version': 'null',
                    'reason': 'no_provider',
                    'logfire.json_schema': '{"type":"object","properties":{"name":{},"targeting_key":{"type":"null"},"attributes":{"type":"object"},"value":{},"label":{"type":"null"},"version":{"type":"null"},"reason":{}}}',
                },
            }
        ]
    )


async def test_baggage_propagates_to_run_and_child_spans(capfire: CaptureLogfire) -> None:
    # `Instrumentation` produces the agent run / model request / tool spans; `ManagedPrompt`
    # runs outermost so its `logfire.variables.prompt__baggage_slug` baggage lands on all of them.
    # The resolution span itself precedes the open baggage context, so it carries no baggage attribute.
    agent = Agent(
        TestModel(),
        capabilities=[ManagedPrompt('baggage_slug', default=DEFAULT), Instrumentation()],
    )

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    await agent.run('hello')

    assert span_attributes(capfire) == snapshot(
        [
            {
                'name': 'Resolve variable prompt__baggage_slug',
                'attributes': {
                    'code.filepath': '_managed_prompt.py',
                    'code.function': 'wrap_run',
                    'targeting_key': 'null',
                    'logfire.msg_template': 'Resolve variable prompt__baggage_slug',
                    'logfire.msg': 'Resolve variable prompt__baggage_slug',
                    'logfire.span_type': 'span',
                    'name': 'prompt__baggage_slug',
                    'value': '"You are a helpful assistant."',
                    'label': 'null',
                    'version': 'null',
                    'reason': 'no_provider',
                    'logfire.json_schema': '{"type":"object","properties":{"name":{},"targeting_key":{"type":"null"},"attributes":{"type":"object"},"value":{},"label":{"type":"null"},"version":{"type":"null"},"reason":{}}}',
                },
            },
            {
                'name': 'chat test',
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'test',
                    'gen_ai.system': 'test',
                    'gen_ai.request.model': 'test',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.tool.definitions': '[{"type":"function","name":"noop","parameters":{"additionalProperties":false,"properties":{},"type":"object"}}]',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat test',
                    'logfire.variables.prompt__baggage_slug': '<code_default>',
                    'gen_ai.input.messages': '[{"role":"user","parts":[{"type":"text","content":"hello"}]}]',
                    'gen_ai.output.messages': '[{"role":"assistant","parts":[{"type":"tool_call","id":"pyd_ai_tool_call_id__noop","name":"noop","arguments":{}}]}]',
                    'gen_ai.system_instructions': '[{"type":"text","content":"You are a helpful assistant."}]',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.input.messages":{"type":"array"},"gen_ai.output.messages":{"type":"array"},"gen_ai.system_instructions":{"type":"array"},"model_request_parameters":{"type":"object"}}}',
                    'gen_ai.usage.input_tokens': 51,
                    'gen_ai.usage.output_tokens': 2,
                    'gen_ai.response.model': 'test',
                },
            },
            {
                'name': 'execute_tool noop',
                'attributes': {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': 'noop',
                    'gen_ai.tool.call.id': 'pyd_ai_tool_call_id__noop',
                    'gen_ai.tool.call.arguments': '{}',
                    'gen_ai.agent.name': 'agent',
                    'logfire.msg': 'running tool: noop',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.call.arguments":{"type":"object"},"gen_ai.tool.call.result":{"type":"object"},"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
                    'logfire.span_type': 'span',
                    'logfire.variables.prompt__baggage_slug': '<code_default>',
                    'gen_ai.tool.call.result': 'ok',
                },
            },
            {
                'name': 'chat test',
                'attributes': {
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.provider.name': 'test',
                    'gen_ai.system': 'test',
                    'gen_ai.request.model': 'test',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.tool.definitions': '[{"type":"function","name":"noop","parameters":{"additionalProperties":false,"properties":{},"type":"object"}}]',
                    'logfire.span_type': 'span',
                    'logfire.msg': 'chat test',
                    'logfire.variables.prompt__baggage_slug': '<code_default>',
                    'gen_ai.input.messages': '[{"role":"user","parts":[{"type":"text","content":"hello"}]},{"role":"assistant","parts":[{"type":"tool_call","id":"pyd_ai_tool_call_id__noop","name":"noop","arguments":{}}]},{"role":"user","parts":[{"type":"tool_call_response","id":"pyd_ai_tool_call_id__noop","name":"noop","result":"ok"}]}]',
                    'gen_ai.output.messages': '[{"role":"assistant","parts":[{"type":"text","content":"{\\"noop\\":\\"ok\\"}"}]}]',
                    'gen_ai.system_instructions': '[{"type":"text","content":"You are a helpful assistant."}]',
                    'logfire.json_schema': '{"type":"object","properties":{"gen_ai.input.messages":{"type":"array"},"gen_ai.output.messages":{"type":"array"},"gen_ai.system_instructions":{"type":"array"},"model_request_parameters":{"type":"object"}}}',
                    'gen_ai.usage.input_tokens': 52,
                    'gen_ai.usage.output_tokens': 6,
                    'gen_ai.response.model': 'test',
                },
            },
            {
                'name': 'invoke_agent agent',
                'attributes': {
                    'model_name': 'test',
                    'agent_name': 'agent',
                    'gen_ai.agent.name': 'agent',
                    'gen_ai.operation.name': 'invoke_agent',
                    'logfire.msg': 'agent run',
                    'logfire.span_type': 'span',
                    'logfire.variables.prompt__baggage_slug': '<code_default>',
                    'final_result': '{"noop":"ok"}',
                    'gen_ai.aggregated_usage.input_tokens': 103,
                    'gen_ai.aggregated_usage.output_tokens': 8,
                    'pydantic_ai.all_messages': '[{"role":"user","parts":[{"type":"text","content":"hello"}]},{"role":"assistant","parts":[{"type":"tool_call","id":"pyd_ai_tool_call_id__noop","name":"noop","arguments":{}}]},{"role":"user","parts":[{"type":"tool_call_response","id":"pyd_ai_tool_call_id__noop","name":"noop","result":"ok"}]},{"role":"assistant","parts":[{"type":"text","content":"{\\"noop\\":\\"ok\\"}"}]}]',
                    'gen_ai.system_instructions': '[{"type":"text","content":"You are a helpful assistant."}]',
                    'logfire.json_schema': '{"type":"object","properties":{"pydantic_ai.all_messages":{"type":"array"},"gen_ai.system_instructions":{"type":"array"},"final_result":{"type":"object"}}}',
                },
            },
        ]
    )


async def test_resolved_once_per_run_across_multiple_model_requests() -> None:
    capability = ManagedPrompt('once_slug', default=DEFAULT)
    agent = Agent(TestModel(), capabilities=[capability])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        result = await agent.run('hello')

    # TestModel issues one request to call the tool and another for the final output,
    # so instructions render twice, but the variable is resolved exactly once.
    assert len(instructions_seen(result.all_messages())) == 2
    assert spy.call_count == 1


async def test_label_and_callable_targeting_and_attributes() -> None:
    capability = ManagedPrompt(
        'targeting_slug',
        default=DEFAULT,
        label='production',
        targeting_key=lambda ctx: f'run:{ctx.run_step}',
        attributes=lambda ctx: {'tier': 'enterprise'},
    )
    agent = Agent(TestModel(), capabilities=[capability])

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hello')

    spy.assert_called_once_with(
        targeting_key='run:0',
        attributes={'tier': 'enterprise'},
        label='production',
    )


async def test_static_targeting_and_attributes() -> None:
    capability = ManagedPrompt(
        'static_slug',
        default=DEFAULT,
        targeting_key='tenant-123',
        attributes={'tier': 'free'},
    )
    agent = Agent(TestModel(), capabilities=[capability])

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:
        await agent.run('hello')

    spy.assert_called_once_with(
        targeting_key='tenant-123',
        attributes={'tier': 'free'},
        label=None,
    )


def test_instructions_none_outside_run() -> None:
    capability: ManagedPrompt[None] = ManagedPrompt('outside_slug', default=DEFAULT)
    instructions = capability.get_instructions()
    ctx = RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )

    # Outside of `wrap_run` nothing has been resolved, so no instructions are contributed.
    assert capability.resolved is None
    assert instructions(ctx) is None


async def test_render_template_fills_from_deps() -> None:
    @dataclass
    class Deps:
        name: str

    capability: ManagedPrompt[Deps] = ManagedPrompt('render_slug', default='Hello {{name}}!', render_template=True)
    agent = Agent(TestModel(), deps_type=Deps, capabilities=[capability])

    result = await agent.run('hi', deps=Deps(name='Alice'))

    assert instructions_seen(result.all_messages()) == ['Hello Alice!']


async def test_resolved_property_exposes_active_resolution() -> None:
    capability = ManagedPrompt('exposed_slug', default=DEFAULT)
    agent = Agent(TestModel(), capabilities=[capability])
    captured: list[str | None] = []

    @agent.tool_plain
    def grab() -> str:
        # `resolved` exposes the full ResolvedVariable for the active run.
        resolved = capability.resolved
        captured.append(resolved.value if resolved is not None else None)
        return 'ok'

    await agent.run('hello')

    assert captured == [DEFAULT]
    # The resolution is cleared once the run completes.
    assert capability.resolved is None


def _remote_prompt_config() -> VariablesConfig:
    return VariablesConfig(
        variables={
            'prompt__remote_slug': VariableConfig(
                name='prompt__remote_slug',
                labels={'production': LabeledValue(version=2, serialized_value='"You are the PRODUCTION prompt."')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )


async def test_provider_backed_resolution_uses_remote_value_and_label(capfire: CaptureLogfire) -> None:
    with _variables_provider_configured(capfire, _remote_prompt_config()):
        agent = Agent(TestModel(), capabilities=[ManagedPrompt('remote_slug', default='fallback', label='production')])

        result = await agent.run('hello')

    # The remote value -- not the code default -- backs the instructions.
    assert instructions_seen(result.all_messages()) == ['You are the PRODUCTION prompt.']

    spans = capfire.exporter.exported_spans_as_dict()
    resolution = next(s for s in spans if s['attributes'].get('logfire.msg') == 'Resolve variable prompt__remote_slug')
    assert resolution['attributes']['reason'] == 'resolved'
    assert resolution['attributes']['value'] == '"You are the PRODUCTION prompt."'
    assert resolution['attributes']['label'] == 'production'


async def test_provider_backed_resolution_tags_v1_instrumentation_spans(capfire: CaptureLogfire) -> None:
    with _variables_provider_configured(capfire, _remote_prompt_config()):
        agent = Agent(
            TestModel(),
            capabilities=[ManagedPrompt('remote_slug', default='fallback', label='production'), Instrumentation()],
        )

        await agent.run('hello')

    spans = capfire.exporter.exported_spans_as_dict()
    # Child spans are tagged with the resolved label via baggage.
    tagged = {s['name'] for s in spans if s['attributes'].get('logfire.variables.prompt__remote_slug') == 'production'}
    assert {'invoke_agent agent', 'chat test'} <= tagged


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='prompt__instance_conflict', type=str, default=DEFAULT)
    with pytest.warns(UserWarning, match='is ignored when `name` is a `Variable`'):
        ManagedPrompt(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)
