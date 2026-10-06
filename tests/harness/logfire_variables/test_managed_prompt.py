"""Tests for the `ManagedPrompt` capability (source package `pydantic_ai_harness.logfire`).

Shared fixtures (`anyio_backend`, Logfire configuration) live in `conftest.py`, which also explains
why the directory is named `logfire_variables` rather than `logfire`. Shared helpers live in
`_helpers.py`, and the variable-naming contract common to all managed-variable capabilities is
covered in `test_managed_variable.py`. This module focuses on `ManagedPrompt` resolving a prompt
into the agent's instructions (including template rendering) and the resolution span it records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import logfire
import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire
from logfire.variables import LabeledValue, Rollout, VariableConfig, VariablesConfig
from logfire.variables.abstract import NoOpVariableProvider

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import ManagedPrompt
from pydantic_ai_harness.logfire import ManagedPrompt as ManagedPromptFromPackage

from ._helpers import variables_provider

DEFAULT = 'You are a helpful assistant.'

# pydantic-ai 2.0.0 renamed the agent run span from `agent run` to `invoke_agent agent` and the tool
# span from `running tool` to `execute_tool noop`. The floor has been `pydantic-ai-slim>=2.14.1` since
# then, so the 1.x branch these were guarded on could no longer be reached -- and the guard read the
# version through `importlib.metadata`, which reports the wrong package's version while this PR
# resolves pydantic-ai from a git source, turning a dead branch into a failing one.
_AGENT_SPAN = 'invoke_agent agent'
_TOOL_SPAN = 'execute_tool noop'


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
_VOLATILE_SPAN_ATTRIBUTES = (
    'attributes',
    'code.lineno',
    'gen_ai.conversation.id',
    'gen_ai.agent.call.id',
    'logfire.metrics',
)


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


def test_no_variable_provider_is_configured() -> None:
    """The resolution tests in this module rely on the no-provider fallback; assert it.

    With a Logfire credential in `os.environ`, `logfire.configure()` lazily resolves a
    `LogfireRemoteVariableProvider` that writes to the real Logfire API from background
    threads, and the resolution tests then fail on span contents far from the cause.
    `conftest.py` removes ambient credentials; this test fails at the boundary if a
    credential still reaches the provider.
    """
    provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
    assert isinstance(provider, NoOpVariableProvider)


def test_slug_becomes_prompt_variable_name() -> None:
    capability = ManagedPrompt('support_agent', default=DEFAULT)
    assert capability._variable.name == 'prompt__support_agent'  # pyright: ignore[reportPrivateUsage]


def test_slug_requires_default() -> None:
    with pytest.raises(TypeError, match='`default` is required'):
        ManagedPrompt('no_default_slug')


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

    with capability._variable.override('Be terse.'):  # pyright: ignore[reportPrivateUsage]
        result = await agent.run('hello')

    assert instructions_seen(result.all_messages()) == ['Be terse.']


async def test_records_variable_resolution_span(capfire: CaptureLogfire) -> None:
    agent = Agent(TestModel(), capabilities=[ManagedPrompt('span_slug', default=DEFAULT)])

    await agent.run('hello')

    # Without `Instrumentation` the only span is the one Logfire records for resolving the
    # prompt variable -- the resolved value, label, version, and reason are captured as attributes.
    spans = span_attributes(capfire)
    # The `reason` for a no-provider resolution is worded differently across logfire versions
    # ('no_provider' before 4.37, 'code_default' after), so assert it separately and drop it from the
    # structural snapshot below.
    assert len(spans) == 1
    assert spans[0]['attributes'].pop('reason') in ('code_default', 'no_provider')
    assert spans == snapshot(
        [
            {
                'name': 'Resolve variable prompt__span_slug',
                'attributes': {
                    'code.filepath': '_managed_variable.py',
                    'code.function': '_resolve',
                    'targeting_key': 'null',
                    'logfire.msg_template': 'Resolve variable prompt__span_slug',
                    'logfire.msg': 'Resolve variable prompt__span_slug',
                    'logfire.span_type': 'span',
                    'name': 'prompt__span_slug',
                    'value': '"You are a helpful assistant."',
                    'label': 'null',
                    'version': 'null',
                    'logfire.json_schema': '{"type":"object","properties":{"name":{},"targeting_key":{"type":"null"},"attributes":{"type":"object"},"value":{},"label":{"type":"null"},"version":{"type":"null"},"reason":{}}}',
                },
            }
        ]
    )


async def test_baggage_propagates_to_run_and_child_spans(capfire: CaptureLogfire) -> None:
    # `Instrumentation` produces the agent-run / model-request / tool spans; `ManagedPrompt` runs
    # outermost, so its `logfire.variables.prompt__baggage_slug` baggage lands on all of them. The
    # resolution span itself precedes the open baggage context, so it carries no baggage attribute.
    # Asserted by baggage presence per span (not a full attribute snapshot) so it stays robust across
    # pydantic-ai instrumentation changes -- only the span *names* differ between the 1.x and 2.x lines.
    agent = Agent(
        TestModel(),
        capabilities=[ManagedPrompt('baggage_slug', default=DEFAULT), Instrumentation()],
    )

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    await agent.run('hello')

    spans = span_attributes(capfire)
    baggage_key = 'logfire.variables.prompt__baggage_slug'

    # The resolution span runs before the baggage context opens, so it is untagged.
    resolution = next(s for s in spans if s['name'] == 'Resolve variable prompt__baggage_slug')
    assert baggage_key not in resolution['attributes']

    # Every run / model-request / tool span the run produces is tagged with the resolved value.
    tagged = {s['name'] for s in spans if s['attributes'].get(baggage_key) == '<code_default>'}
    assert {_AGENT_SPAN, _TOOL_SPAN, 'chat test'} <= tagged


async def test_resolved_once_per_run_across_multiple_model_requests() -> None:
    capability = ManagedPrompt('once_slug', default=DEFAULT)
    agent = Agent(TestModel(), capabilities=[capability])

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:  # pyright: ignore[reportPrivateUsage]
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

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:  # pyright: ignore[reportPrivateUsage]
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

    with patch.object(capability._variable, 'get', wraps=capability._variable.get) as spy:  # pyright: ignore[reportPrivateUsage]
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


async def test_provider_backed_resolution_uses_remote_value_and_label(capfire: CaptureLogfire) -> None:
    config = VariablesConfig(
        variables={
            'prompt__remote_slug': VariableConfig(
                name='prompt__remote_slug',
                labels={'production': LabeledValue(version=2, serialized_value='"You are the PRODUCTION prompt."')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(
            TestModel(),
            capabilities=[ManagedPrompt('remote_slug', default='fallback', label='production'), Instrumentation()],
        )

        result = await agent.run('hello')

    # The remote value -- not the code default -- backs the instructions.
    assert instructions_seen(result.all_messages()) == ['You are the PRODUCTION prompt.']

    spans = capfire.exporter.exported_spans_as_dict()
    resolution = next(s for s in spans if s['attributes'].get('logfire.msg') == 'Resolve variable prompt__remote_slug')
    assert resolution['attributes']['reason'] == 'resolved'
    assert resolution['attributes']['value'] == '"You are the PRODUCTION prompt."'
    assert resolution['attributes']['label'] == 'production'
    # Child spans are tagged with the resolved label via baggage.
    tagged = {s['name'] for s in spans if s['attributes'].get('logfire.variables.prompt__remote_slug') == 'production'}
    assert {_AGENT_SPAN, 'chat test'} <= tagged


async def test_the_label_survives_a_prompt_name_the_scrubber_matches(capfire: CaptureLogfire) -> None:
    """A prompt's slug is ordinary English, and Logfire's scrubbing matches substrings.

    The label and version ride on every span of the run as `logfire.variables.<name>`, whose key is
    built from the slug -- so a prompt called `session_summary` matched `session` on the *key* and
    every span said `[Scrubbed due to 'session']` where the label belonged. That is the whole
    attribution: "which version of this prompt produced this answer" is what a managed prompt is
    audited by. Keys under `logfire.variables.` are safe in the `logfire` package, and this is the
    shipped capability that needed it.
    """
    config = VariablesConfig(
        variables={
            'prompt__session_summary': VariableConfig(
                name='prompt__session_summary',
                labels={'production': LabeledValue(version=7, serialized_value='"Summarize the session."')},
                rollout=Rollout(labels={'production': 1.0}),
                overrides=[],
            )
        }
    )
    with variables_provider(capfire, config):
        agent = Agent(
            TestModel(),
            capabilities=[ManagedPrompt('session_summary', default='fallback', label='production'), Instrumentation()],
        )

        await agent.run('hello')

    spans = capfire.exporter.exported_spans_as_dict()
    labelled = {
        span['name']: (
            span['attributes'].get('logfire.variables.prompt__session_summary'),
            span['attributes'].get('logfire.variables.prompt__session_summary.version'),
        )
        for span in spans
        if span['name'] in {_AGENT_SPAN, 'chat test'}
    }
    assert labelled == {_AGENT_SPAN: ('production', '7'), 'chat test': ('production', '7')}
    assert all('logfire.scrubbed' not in span['attributes'] for span in spans if span['name'] in labelled)


def test_logfire_instance_with_prebuilt_variable_warns() -> None:
    var = logfire.var(name='prompt__instance_conflict', type=str, default=DEFAULT)
    with pytest.warns(UserWarning, match='is ignored when `name` is a `Variable`'):
        ManagedPrompt(var, logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE)


def test_original_positional_arguments_remain_compatible() -> None:
    capability = ManagedPrompt(
        'support_agent',
        'fallback',
        'production',
        'customer-123',
        {'plan': 'pro'},
        True,
        None,
    )

    assert capability.name == 'support_agent'
    assert capability.default == 'fallback'
    assert capability.label == 'production'
    assert capability.targeting_key == 'customer-123'
    assert capability.attributes == {'plan': 'pro'}
    assert capability.render_template is True
    assert capability.logfire_instance is None
