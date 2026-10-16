from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Any, cast

import logfire
import pytest
from inline_snapshot import snapshot
from logfire._internal.config import LogfireConfig
from logfire.agent_control import (
    MAX_MODEL_FACING_TEXT_LENGTH,
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    ParameterOverride,
    ToolDefinitionOverride,
    build_baseline,
)
from logfire.testing import CaptureLogfire
from logfire.variables import Variable, VariablesConfig

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, Capability, CombinedCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import AgentControl as RootAgentControl
from pydantic_ai_harness.logfire import AgentControl, _agent_control

from ._helpers import (
    Publish,
    advertised,
    capture_tools,
    get_forecast,
    get_weather,
    published_value,
    variables_provider,
    weather_toolset,
)

assert RootAgentControl is AgentControl


def instructions_seen(messages: list[ModelMessage]) -> list[str]:
    return [m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions is not None]


def config_hint(capfire: CaptureLogfire) -> dict[str, Any]:
    """The attributes of the run's config-hint span. `test_config_hint.py` owns its full contract."""
    spans = capfire.exporter.exported_spans_as_dict()
    return next(span['attributes'] for span in spans if span['name'] == 'agent_control_config_hint')


async def test_empty_config_keeps_code_behavior() -> None:
    result = await Agent(TestModel(), instructions='code', capabilities=[AgentControl('empty')]).run('hello')
    assert instructions_seen(result.all_messages()) == ['code']


_UNREADABLE = (
    "`AgentControl` cannot read 'agent__unreadable': its Logfire instance has no variable provider, so the agent "
    'runs as defined in code and nothing published in Logfire applies. '
)
_NO_API_KEY = (
    'Reading Logfire variables needs an API key with the `project:read_variables` scope (a write token cannot '
    'read them), set as `LOGFIRE_API_KEY` for `logfire.configure()` to read, or passed as '
    '`logfire.configure(api_key=...)`.'
)
_UNCONFIGURED_DEFAULT = (
    '`LOGFIRE_API_KEY` is set, but the default Logfire instance has not been configured, and only a configured '
    'instance reads it. Call `logfire.configure()`.'
)
_UNCONFIGURED_GIVEN = (
    '`LOGFIRE_API_KEY` is set, but its Logfire instance has not been configured, and only a configured instance '
    'reads it. Use the instance `logfire.configure()` returned.'
)
_PASS_INSTANCE = (
    ' This capability resolves on the default Logfire instance: if your application configures its own, pass it '
    'as `AgentControl(logfire_instance=...)`.'
)
_CREATE_VARIABLE = (
    ' This variable was created on the default Logfire instance: if your application configures its own, create '
    'the `Variable` on yours.'
)


@pytest.mark.parametrize(
    'api_key,instance,expected',
    [
        # This package configures Logfire without an API key, which is the deployment that holds only a
        # write token.
        pytest.param(False, 'default', _UNREADABLE + _NO_API_KEY + _PASS_INSTANCE, id='no-api-key'),
        # A host application that configures a Logfire instance of its own leaves the default one
        # unconfigured, and an unconfigured instance never reads `LOGFIRE_API_KEY`.
        pytest.param(
            True, 'default', _UNREADABLE + _UNCONFIGURED_DEFAULT + _PASS_INSTANCE, id='unconfigured-default-instance'
        ),
        pytest.param(
            True,
            'default-variable',
            _UNREADABLE + _UNCONFIGURED_DEFAULT + _CREATE_VARIABLE,
            id='variable-on-unconfigured-default-instance',
        ),
        pytest.param(True, 'given', _UNREADABLE + _UNCONFIGURED_GIVEN, id='unconfigured-given-instance'),
        pytest.param(True, 'given-variable', _UNREADABLE + _UNCONFIGURED_GIVEN, id='variable-on-unconfigured-instance'),
    ],
)
async def test_an_instance_that_cannot_read_variables_warns_once(
    monkeypatch: pytest.MonkeyPatch, api_key: bool, instance: str, expected: str
) -> None:
    # Resolving through no provider at all looks identical to "nothing published yet" -- the agent runs
    # as written either way -- so it has to say which it is, and what is missing: once, not every run.
    # Nothing is fetched in any case: an unconfigured instance never reads the key, which is the point.
    unconfigured = logfire.Logfire(config=LogfireConfig())
    if api_key:
        monkeypatch.setenv('LOGFIRE_API_KEY', 'pylf_v1_us_not_a_real_key')
        monkeypatch.setenv('LOGFIRE_IGNORE_NO_CONFIG', '1')
        monkeypatch.setattr(logfire, 'DEFAULT_LOGFIRE_INSTANCE', unconfigured)
    if instance == 'default':
        control = AgentControl('unreadable')
    elif instance == 'given':
        control = AgentControl('unreadable', logfire_instance=logfire.Logfire(config=LogfireConfig()))
    else:
        on = unconfigured if instance == 'default-variable' else logfire.Logfire(config=LogfireConfig())
        control = AgentControl(
            Variable('agent__unreadable', type=AgentConfig, default=AgentConfig(), logfire_instance=on)
        )
    agent = Agent(TestModel(), instructions='code', capabilities=[control])
    with pytest.warns(UserWarning) as caught:
        await agent.run('hello')
        result = await agent.run('hello')
    assert [str(w.message) for w in caught] == [expected]
    assert instructions_seen(result.all_messages()) == ['code']


async def test_a_provider_with_nothing_published_stays_quiet(capfire: CaptureLogfire) -> None:
    # An agent nobody has configured yet is the normal state of a new agent, not a misconfiguration.
    agent = Agent(TestModel(), instructions='code', capabilities=[AgentControl('unpublished')])
    with variables_provider(capfire, VariablesConfig(variables={})):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = await agent.run('hello')
    assert caught == []
    assert instructions_seen(result.all_messages()) == ['code']


async def test_managed_instructions_are_appended_not_replaced(publish: Publish) -> None:
    publish('instructions', {'instructions': 'managed'})
    agent = Agent(
        TestModel(),
        instructions='code',
        toolsets=[FunctionToolset[object](instructions='toolset')],
        capabilities=[AgentControl('instructions', label='production')],
    )

    @agent.instructions
    def dynamic(ctx: RunContext[object]) -> str:
        return 'dynamic'

    result = await agent.run('hello')
    # An entry with no `id` adds a block, so everything code-defined still reaches the model. The
    # added block lands at the end of the static run -- last in the prompt as written, and still
    # ahead of the dynamic text a provider cannot cache.
    assert instructions_seen(result.all_messages()) == ['code\n\ntoolset\n\nmanaged\n\ndynamic']


def capture_instructions(seen: list[InstructionPart]) -> FunctionModel:
    """A model that records the instruction blocks it is shown, then ends the run.

    The parts, not the joined string, are what a block-addressing config has to be judged on: only
    they carry the `id` an override addresses and the `dynamic` flag that decides where a block sorts.
    """

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # `None` rather than `[]` is the shape an agent with no instructions at all produces.
        seen.extend(info.model_request_parameters.instruction_parts or [])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


async def run_blocks(capfire: CaptureLogfire, name: str, value: dict[str, Any]) -> list[InstructionPart]:
    """Publish `value` for `agent__<name>`, run an agent with a block of each kind, return what it sent.

    The agent covers every source a managed `id` can address: its own literal (`agent`), one declared
    `@agent.instructions` function (`agent:today`), and a toolset with an `id` (`toolset:weather`).
    """
    seen: list[InstructionPart] = []
    agent = Agent(
        capture_instructions(seen),
        instructions='AGENT: You are a concise checkout assistant.',
        toolsets=[weather_toolset()],
        capabilities=[AgentControl(name, label='production')],
    )

    @agent.instructions(name='today')
    def today(_ctx: RunContext[object]) -> str:
        return 'DYNAMIC: today is Monday.'

    with variables_provider(capfire, published_value(f'agent__{name}', value)):
        await agent.run('hello')
    return seen


def triples(parts: list[InstructionPart]) -> list[tuple[str | None, str, bool]]:
    """Each part as the `(id, text, dynamic)` the model was sent it under.

    The id is rendered to the string a managed config addresses it by, which is the form this
    capability's whole contract is written in.
    """
    return [(str(part.id) if part.id is not None else None, part.content, part.dynamic) for part in parts]


AGENT_BLOCK = ('agent', 'AGENT: You are a concise checkout assistant.', False)
TODAY_BLOCK = ('agent:today', 'DYNAMIC: today is Monday.', True)
TOOLSET_BLOCK = ('toolset:weather', 'TOOLSET: call get_weather first.', True)
ADDED_BRIEF = (None, 'MANAGED: be brief.', False)
"""An added block is unaddressable (nothing in code assembled it) and static (it is the same text on
every request of the run), which is what puts it inside the prefix a provider can cache."""


@pytest.mark.parametrize(
    'name,value,expected',
    [
        pytest.param('blocks_none', {}, [AGENT_BLOCK, TODAY_BLOCK, TOOLSET_BLOCK], id='nothing-managed'),
        pytest.param(
            'blocks_bare',
            {'instructions': 'MANAGED: be brief.'},
            [AGENT_BLOCK, ADDED_BRIEF, TODAY_BLOCK, TOOLSET_BLOCK],
            id='bare-string-adds-a-block',
        ),
        pytest.param(
            'blocks_list',
            {'instructions': ['MANAGED: be brief.', 'MANAGED: cite sources.']},
            [AGENT_BLOCK, ADDED_BRIEF, (None, 'MANAGED: cite sources.', False), TODAY_BLOCK, TOOLSET_BLOCK],
            id='each-added-entry-is-its-own-block',
        ),
        pytest.param(
            'blocks_mixed',
            {'instructions': ['MANAGED: be brief.', {'id': 'agent', 'instructions': 'REMOTE: refund specialist.'}]},
            [
                ('agent', 'REMOTE: refund specialist.', False),
                ADDED_BRIEF,
                TODAY_BLOCK,
                TOOLSET_BLOCK,
            ],
            id='one-list-can-add-and-address',
        ),
        pytest.param(
            'blocks_replace',
            {'instructions': [{'id': 'agent', 'instructions': 'REMOTE: refund specialist.'}]},
            [('agent', 'REMOTE: refund specialist.', False), TODAY_BLOCK, TOOLSET_BLOCK],
            id='replace-the-agent-literal',
        ),
        pytest.param(
            'blocks_drop',
            {'instructions': [{'id': 'agent', 'instructions': None}]},
            [TODAY_BLOCK, TOOLSET_BLOCK],
            id='drop-the-agent-literal',
        ),
    ],
)
async def test_instruction_blocks_the_model_receives(
    capfire: CaptureLogfire, name: str, value: dict[str, Any], expected: list[tuple[str | None, str, bool]]
) -> None:
    # The whole contract in one table: an entry with no `id` adds a block, and an entry with one
    # replaces or drops the block the agent assembled under that key.
    #
    # Added blocks land at the end of the static run -- after the agent's own literal, before the
    # first block the agent recomputes per request -- so published text is last in the prompt as
    # written and still inside the prefix a provider can cache. One part per entry: each entry is one
    # block, and joining them would make two published entries indistinguishable from one.
    assert triples(await run_blocks(capfire, name, value)) == expected


async def test_an_id_nothing_matches_applies_nothing_and_warns_once(capfire: CaptureLogfire) -> None:
    # One config is applied across deployments that need not all install the same toolsets, so an
    # `id` this agent never assembles is not an error by default. It is a place where Logfire shows
    # one thing and the agent does another, though, so it warns -- once per process, not per request.
    with pytest.warns(UserWarning, match=r"addresses instruction block 'toolset:nope', which this request") as caught:
        parts = await run_blocks(
            capfire, 'blocks_unmatched', {'instructions': [{'id': 'toolset:nope', 'instructions': 'never sent'}]}
        )
    assert len(caught) == 1
    assert triples(parts) == [AGENT_BLOCK, TODAY_BLOCK, TOOLSET_BLOCK]


@pytest.mark.parametrize(
    'value',
    [
        pytest.param(
            {'instructions': [{'id': 'agent:today', 'instructions': 'PINNED: it is always Monday.'}]},
            id='replacing-it',
        ),
        pytest.param({'instructions': [{'id': 'toolset:weather', 'instructions': None}]}, id='dropping-it'),
    ],
)
async def test_a_dynamic_block_is_not_addressable(capfire: CaptureLogfire, value: dict[str, Any]) -> None:
    """A block the agent recomputes per request is not the managed config's to change.

    Replacing it pins one rendering forever; dropping it removes the computation. Either way the block
    stops doing what it was written to do, so both are refused and warned about. Note `toolset:weather`
    returns a constant and is flagged dynamic all the same: a function's shape says nothing about
    whether its text came from the run, so an author who wants fixed text addressable contributes an
    `InstructionPart` rather than a function.
    """
    with pytest.warns(UserWarning, match='which the agent recomputes per request'):
        blocks = triples(await run_blocks(capfire, 'blocks_dynamic_refused', value))
    assert blocks == [AGENT_BLOCK, TODAY_BLOCK, TOOLSET_BLOCK]


async def test_an_entry_without_instructions_does_not_drop_its_block(capfire: CaptureLogfire) -> None:
    # `text` is not a key the contract knows, so this entry validates with `instructions` unset --
    # which the contract reads as `null`, the way a block is dropped. One typo would take the agent's
    # whole prompt with it. Only an explicit `null` drops a block, so this keeps the code-defined text,
    # says why, and leaves the rest of the section applied, an explicit `null` beside it included.
    seen: list[InstructionPart] = []
    agent = Agent(
        capture_instructions(seen),
        instructions=[
            'AGENT: You are a concise checkout assistant.',
            InstructionPart(content='AGENT: Confirm the total before refunding.', name='rules'),
        ],
        capabilities=[AgentControl('blocks_typo', label='production')],
    )
    published = {
        'instructions': [
            {'id': 'agent', 'text': 'REMOTE: refund specialist.'},
            {'id': 'agent:rules', 'instructions': None},
            'MANAGED: be brief.',
        ]
    }
    with variables_provider(capfire, published_value('agent__blocks_typo', published)):
        with pytest.warns(UserWarning) as caught:
            await agent.run('hello')
    assert [str(w.message) for w in caught] == snapshot(
        [
            'Managed instruction entry for block \'agent\' has no `instructions` key, so it is not applied and the block keeps its code-defined text. Publish `"instructions": null` to drop the block, or the replacement text to replace it; a misspelled key such as `text` reads as a missing one.'
        ]
    )
    assert triples(seen) == [AGENT_BLOCK, ADDED_BRIEF]


async def test_an_override_never_moves_the_static_prefix(capfire: CaptureLogfire) -> None:
    # `dynamic` decides which side of the provider's cacheable prefix a block sorts on, and an override
    # rewrites text in place without touching it. So overriding the agent's static literal leaves every
    # block's key and flag exactly where code put them: the cache boundary does not move, which is the
    # whole reason the flag is carried over rather than recomputed.
    baseline = await run_blocks(capfire, 'prefix_baseline', {})
    overridden = await run_blocks(
        capfire,
        'prefix_overridden',
        {'instructions': [{'id': 'agent', 'instructions': 'REMOTE: refund specialist.'}]},
    )
    assert [(part.id, part.dynamic) for part in overridden] == [(part.id, part.dynamic) for part in baseline]
    assert [part.content for part in overridden] == [
        'REMOTE: refund specialist.',
        'DYNAMIC: today is Monday.',
        'TOOLSET: call get_weather first.',
    ]


async def test_a_dropped_block_does_not_shift_what_a_later_entry_replaces(capfire: CaptureLogfire) -> None:
    # Two blocks the agent assembles under different keys, one dropped and the other replaced. The
    # replacement has to land on the block that was addressed rather than on whatever moved up into
    # its position, which is the one thing that can go wrong when the applied blocks are matched back
    # onto the parts they came from.
    seen: list[InstructionPart] = []
    agent = Agent(
        capture_instructions(seen),
        instructions=[
            'AGENT: You are a concise checkout assistant.',
            InstructionPart(content='AGENT: Confirm the total before refunding.', name='rules'),
        ],
        capabilities=[AgentControl('blocks_drop_then_replace', label='production')],
    )
    published = {
        'instructions': [
            {'id': 'agent', 'instructions': None},
            {'id': 'agent:rules', 'instructions': 'REMOTE: refund rules.'},
            'MANAGED: be brief.',
        ]
    }
    with variables_provider(capfire, published_value('agent__blocks_drop_then_replace', published)):
        await agent.run('hello')
    assert triples(seen) == [('agent:rules', 'REMOTE: refund rules.', False), ADDED_BRIEF]


async def test_two_entries_addressing_the_same_block_keep_the_first(capfire: CaptureLogfire) -> None:
    # A hand-edited value (or a UI bug) can name one `id` twice. Keeping the first makes the run
    # predictable and names the entry that lost, rather than letting JSON ordering decide silently.
    with pytest.warns(UserWarning, match=r"names instruction id 'agent' more than once") as caught:
        parts = await run_blocks(
            capfire,
            'blocks_duplicate',
            {'instructions': [{'id': 'agent', 'instructions': 'FIRST'}, {'id': 'agent', 'instructions': 'SECOND'}]},
        )
    # Resolved once per run and applied on every request, but warned about once.
    assert len(caught) == 1
    assert triples(parts) == [('agent', 'FIRST', False), TODAY_BLOCK, TOOLSET_BLOCK]


async def test_overrides_on_an_agent_with_no_instructions_reach_nothing(capfire: CaptureLogfire) -> None:
    # An agent that sends no instructions at all has nothing to address, so the entry is reported
    # like any other unmatched `id` -- and `on_unmatched='ignore'` is the way to say that is expected.
    seen: list[InstructionPart] = []
    published = {'instructions': [{'id': 'agent', 'instructions': 'never sent'}]}
    with variables_provider(capfire, published_value('agent__no_blocks', published)):
        with pytest.warns(UserWarning, match=r"addresses instruction block 'agent', which this request"):
            result = await Agent(
                capture_instructions(seen), capabilities=[AgentControl('no_blocks', label='production')]
            ).run('hello')
        with warnings.catch_warnings(record=True) as caught:
            quiet = AgentControl('no_blocks', label='production', on_unmatched='ignore')
            await Agent(capture_instructions(seen), capabilities=[quiet]).run('hello')
    assert caught == []
    assert seen == []
    assert instructions_seen(result.all_messages()) == []


async def test_an_added_block_reaches_an_agent_with_no_instructions_of_its_own(capfire: CaptureLogfire) -> None:
    # An agent that assembles nothing has no parts list for a managed block to be added to, so the
    # request goes out with one the capability created. The joined prompt is what the model is sent,
    # so the message history has to carry it too.
    seen: list[InstructionPart] = []
    published = {'instructions': 'MANAGED: be brief.'}
    with variables_provider(capfire, published_value('agent__adds_to_nothing', published)):
        result = await Agent(
            capture_instructions(seen), capabilities=[AgentControl('adds_to_nothing', label='production')]
        ).run('hello')
    assert triples(seen) == [ADDED_BRIEF]
    assert instructions_seen(result.all_messages()) == ['MANAGED: be brief.']


async def test_added_blocks_render_placeholders_against_deps(publish: Publish) -> None:
    @dataclass
    class Deps:
        city: str

    # Every added block is rendered, so a `{{...}}` placeholder works the same in a list entry as it
    # does in a bare string. An addressed block is deliberately left alone: its text replaces
    # something the agent assembled, and that text was never templated either.
    publish(
        'render',
        AgentConfig(
            instructions=[
                'Serve {{city}}.',
                InstructionBlock(instructions='Be brief.'),
                InstructionBlock(id='agent', instructions='Base for {{city}}.'),
            ]
        ),
    )
    seen: list[InstructionPart] = []
    capability: AgentControl[Deps] = AgentControl('render', label='production', render_template=True)
    agent = Agent(capture_instructions(seen), instructions='code', deps_type=Deps, capabilities=[capability])
    result = await agent.run('hello', deps=Deps(city='Paris'))
    assert instructions_seen(result.all_messages()) == ['Base for {{city}}.\n\nServe Paris.\n\nBe brief.']
    # A rendered block is this run's answer rather than fixed text, so it is contributed as dynamic
    # and sorts after the static prompt, where Pydantic AI puts every other `TemplateStr`.
    assert triples(seen) == [
        ('agent', 'Base for {{city}}.', False),
        (None, 'Serve Paris.', True),
        (None, 'Be brief.', True),
    ]


async def test_tool_definition_patches(publish: Publish) -> None:
    seen: list[ToolDefinition] = []
    publish(
        'tools',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(
                    name='get_weather',
                    description='Managed.',
                    parameters={'city': ParameterOverride(description='Managed city.')},
                )
            ]
        ),
    )
    capability = AgentControl('tools', label='production')
    await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert advertised(seen) == {'get_weather': 'Managed.'}
    assert seen[0].parameters_json_schema['properties']['city']['description'] == 'Managed city.'


async def test_settings_lower_by_canonical_key_and_ignore_the_rest(publish: Publish) -> None:
    """Only the canonical keys reach the request; anything else in the section is reported, not forwarded.

    A provider-specific key (`openai_temperature`) and a nested escape hatch (`provider_options`) are
    exactly what an SDK for another framework could not lower, so the contract has neither. Forwarding
    them anyway would make this SDK apply a value the schema never promised, so each is dropped and
    named once per process, at the point the settings are applied.
    """
    seen: list[dict[str, object]] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    publish(
        'settings',
        {
            'settings': {
                'temperature': 0.2,
                'top_p': 0.4,
                'provider_options': {'openai': {'temperature': 0.8}},
                'openai_temperature': 0.6,
            }
        },
    )
    capability = AgentControl('settings', label='production')
    with pytest.warns(UserWarning, match='which this version of the Agent Control contract has no') as caught:
        await Agent(
            FunctionModel(capture_settings),
            model_settings={'temperature': 0.1, 'top_k': 3},
            capabilities=[capability],
        ).run('hello', model_settings={'temperature': 0.9})
    assert sorted(str(warning.message).split(',')[0] for warning in caught) == [
        "Managed agent config sets 'openai_temperature'",
        "Managed agent config sets 'provider_options'",
    ]
    assert seen == [{'temperature': 0.9, 'top_k': 3, 'top_p': 0.4}]


async def test_a_published_config_cannot_crash_a_run_under_an_error_filter(publish: Publish) -> None:
    """`docs/agent-control.md`: "a published config can never crash a run". Asserted where it can break.

    Every other test here reads its warnings through `pytest.warns`, which captures them, so none of
    them can tell whether a drop is able to escalate. Under `-W error` -- or this suite's own
    `filterwarnings = ['error']`, which is how a user most plausibly meets it -- `warnings.warn`
    raises. A validation drop is warned from inside `Variable.get`, so that raise was caught by
    logfire's resolution fallback and reported as "could not read the value": every section that did
    parse stopped applying because one setting had the wrong type, and then the fallback's own warning
    raised on top and reached the run.

    Fixed in the contract (pydantic/logfire#2389) rather than here, since every Agent Control SDK
    inherits it, but the claim is this package's so the test belongs here too. Being strict about
    unmatched entries is `on_unmatched='error'`, which is unaffected.
    """
    seen: list[dict[str, object]] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    # `'0.4'` is a string where the schema says number: dropped rather than coerced, with a warning.
    # Settings alone carry the point -- a managed `model` would need a provider credential to infer,
    # which is a different failure and would hide this one.
    publish('nocrash', {'settings': {'temperature': '0.4', 'max_tokens': 2048}})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = await Agent(
            FunctionModel(capture_settings), capabilities=[AgentControl('nocrash', label='production')]
        ).run('hello')

    assert result.output == 'done'
    # The surviving sections still applied: one bad field costs that field, not the whole config.
    assert seen == [{'max_tokens': 2048}], 'one wrong-typed setting un-managed the rest of the config'


async def test_an_unknown_managed_model_cannot_crash_a_run_under_an_error_filter(publish: Publish) -> None:
    """The same guarantee on this package's own warning path, which is a separate one.

    The drop above is reported by the contract's parser during resolution. This one is reported here,
    at apply time, when a published `model` string does not name a model: the run keeps the code model
    and says so. That warning goes through this module's own `_warn_drop`, so it escalated under an
    error filter independently of the contract's, and took the run down -- for a config whose only
    fault was naming a model this deployment does not have.

    An unresolvable provider rather than a real one with no key: `infer_model` refuses it without
    reaching for a credential, so the test asserts the drop rather than an environment.
    """
    seen: list[dict[str, object]] = []

    def capture_model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    publish('unknownmodel', {'model': 'nonesuch:whatever', 'settings': {'max_tokens': 128}})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = await Agent(
            FunctionModel(capture_model), capabilities=[AgentControl('unknownmodel', label='production')]
        ).run('hello')

    assert result.output == 'done'
    # The run reached the code-defined model rather than raising, and the section that did parse
    # still applied -- one unusable field costs that field.
    assert seen == [{'max_tokens': 128}]


async def test_every_canonical_setting_reaches_the_request(publish: Publish) -> None:
    """`AgentControl` declares it can lower every canonical setting; this is that claim, key by key.

    The declaration is what the contract reports an unapplied key against and what the Logfire editor
    greys a field out on, so "all of them" has to be true rather than convenient. The keys are written
    out here rather than read off the contract, so a key the contract gains fails this test -- with the
    `ModelSettings` field it needs, or with a narrower declaration -- instead of being assumed.
    """
    published = {
        'max_tokens': 128,
        'temperature': 0.2,
        'top_p': 0.4,
        'top_k': 3,
        'seed': 7,
        'presence_penalty': 0.1,
        'frequency_penalty': 0.2,
        'parallel_tool_calls': False,
        'timeout': 30.0,
        'stop_sequences': ['STOP'],
        'thinking': 'low',
    }
    assert set(published) == set(AgentConfigSettings.model_fields)
    seen: list[dict[str, object]] = []
    thinking: list[object] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        # Pydantic AI resolves the unified `thinking` setting against the model's profile and carries
        # it on the request parameters instead, so this is where a managed one arrives. A model whose
        # profile does not support thinking drops it, exactly as it drops one written in code: what a
        # model does with a forwarded setting is the model's business and not this capability's.
        thinking.append(info.model_request_parameters.thinking)
        return ModelResponse(parts=[TextPart('done')])

    publish('all_settings', {'settings': published})
    with warnings.catch_warnings(record=True) as caught:
        await Agent(
            FunctionModel(capture_settings, profile=ModelProfile(supports_thinking=True)),
            capabilities=[AgentControl('all_settings', label='production')],
        ).run('hello')
    assert caught == []
    assert thinking == ['low']
    assert seen == [{key: value for key, value in published.items() if key != 'thinking'}]


def test_prebuilt_variable() -> None:
    variable = Variable(
        'agent__prebuilt',
        type=AgentConfig,
        default=AgentConfig(model='test'),
        logfire_instance=logfire.DEFAULT_LOGFIRE_INSTANCE,
    )
    assert AgentControl(variable)._variable is variable  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('control_first', [True, False])
async def test_managed_values_beat_another_capability(control_first: bool, publish: Publish) -> None:
    """A published value outranks every other capability's, whichever order they were registered in.

    Both directions are asserted because that order-independence is the whole point of the innermost
    companion `for_agent` binds: contributions merge left to right, so a capability running outermost
    -- which `AgentControl` must, to keep its baggage around the run span -- always loses on its own.
    """
    seen: list[dict[str, object]] = []

    def capture_settings(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(parts=[TextPart('done')])

    class Opinionated(Capability[object]):
        """Any other capability contributing a model and settings of its own."""

        def get_model(self) -> TestModel:
            return TestModel(custom_output_text='the other capability')

        def get_model_settings(self) -> ModelSettings:
            return ModelSettings(temperature=0.9, top_k=5)

    def order(control: AgentControl[object]) -> list[AbstractCapability[object]]:
        return [control, Opinionated()] if control_first else [Opinionated(), control]

    # The managed `test` model answers, so the other capability's model never ran.
    publish('contested_model', AgentConfig(model='test'))
    model_control: AgentControl[object] = AgentControl('contested_model', label='production')
    result = await Agent(None, capabilities=order(model_control)).run('hello')
    assert result.output.startswith('success')

    # Settings still merge per key: the managed `temperature` wins, the uncontested `top_k` survives.
    publish('contested_settings', AgentConfig(settings=AgentConfigSettings(temperature=0.2)))
    settings_control: AgentControl[object] = AgentControl('contested_settings', label='production')
    await Agent(FunctionModel(capture_settings), capabilities=order(settings_control)).run('hello')
    assert seen == [{'temperature': 0.2, 'top_k': 5}]


def test_explicit_capability_id_is_preserved() -> None:
    assert AgentControl('explicit_id', id='custom').id == 'custom'


async def test_oversized_code_instructions_do_not_affect_run(capfire: CaptureLogfire) -> None:
    """The length bound is about what a managed value may add, not about what the agent already says.

    An agent whose own instructions exceed it has a block too big to describe, not a broken run, so
    the baseline leaves that block out and says so. Nothing about describing the agent may raise into
    a request, and the rest of the snapshot is still worth reporting.
    """
    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(
            TestModel(),
            instructions=[
                'x' * (MAX_MODEL_FACING_TEXT_LENGTH + 1),
                InstructionPart(content='AGENT: and something publishable.', name='rest'),
            ],
            capabilities=[AgentControl('oversized_code')],
        )
        with pytest.warns(UserWarning, match='leaving it out of the published baseline'):
            result = await agent.run('hello')
    assert result.output.startswith('success')
    assert json.loads(config_hint(capfire)['agent_control.baseline'])['instructions'] == [
        {'id': 'agent:rest', 'instructions': 'AGENT: and something publishable.', 'dynamic': False}
    ]


async def test_applied_sections_baggage(publish: Publish) -> None:
    seen: list[object] = []

    def inspect_baggage() -> str:
        seen.append(logfire.get_baggage().get('logfire.managed.applied_sections'))
        return 'ok'

    publish('baggage', AgentConfig(instructions='managed', settings=AgentConfigSettings(temperature=0.2)))
    capability = AgentControl('baggage', label='production')
    await Agent(TestModel(), tools=[inspect_baggage], capabilities=[capability]).run('hello')
    assert seen == ['instructions,settings']


async def test_empty_config_has_no_applied_sections_baggage() -> None:
    seen: list[object] = []

    def inspect_baggage() -> str:
        seen.append(logfire.get_baggage().get('logfire.managed.applied_sections'))
        return 'ok'

    await Agent(TestModel(), tools=[inspect_baggage], capabilities=[AgentControl('empty_baggage')]).run('hello')
    assert seen == [None]


async def test_rename_round_trip_preserves_original_context_name(publish: Publish) -> None:
    calls = 0
    context_names: list[str | None] = []

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            assert [tool.name for tool in info.function_tools] == ['weather_now']
            return ModelResponse(parts=[ToolCallPart('weather_now', {'city': 'Paris'}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    def weather(ctx: RunContext[object], city: str) -> str:
        context_names.append(ctx.tool_name)
        return city

    publish('rename', AgentConfig(tool_definitions=[ToolDefinitionOverride(name='weather', new_name='weather_now')]))
    capability = AgentControl('rename', label='production')
    await Agent(FunctionModel(model), tools=[weather], capabilities=[capability]).run('hello')
    assert context_names == ['weather']


async def test_rename_collision_warns_and_keeps_other_patches(publish: Publish) -> None:
    seen: list[ToolDefinition] = []
    calls = 0

    def first() -> str:
        return 'first'

    def second() -> str:  # pragma: no cover - advertised only
        return 'second'

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        seen.extend(info.function_tools)
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('first', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    publish(
        'collision',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='first', new_name='second', description='Managed first.'),
            ]
        ),
    )
    capability = AgentControl('collision', label='production')
    with pytest.warns(UserWarning, match='already advertised'):
        await Agent(FunctionModel(model), tools=[first, second], capabilities=[capability]).run('hello')
    assert advertised(seen[:2]) == {'first': 'Managed first.', 'second': None}


async def test_a_patch_on_a_parameter_the_tool_does_not_have_is_reported(publish: Publish) -> None:
    # A parameter is part of the tool's code-defined shape, so a patch naming one the tool does not
    # have is the tool having changed. From the Logfire UI that looks exactly like a patch that
    # applied, which is why it goes through `on_unmatched` like every other entry that reached
    # nothing, and why the tool itself is untouched.
    seen: list[ToolDefinition] = []
    publish(
        'unknown_parameter',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(
                    name='get_weather', parameters={'missing': ParameterOverride(description='ignored')}
                ),
            ]
        ),
    )
    capability = AgentControl('unknown_parameter', label='production')
    with pytest.warns(UserWarning, match=r"patches parameter 'missing' of tool 'get_weather'"):
        await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert advertised(seen) == {'get_weather': None}
    assert get_weather('Paris') == 'sunny in Paris'


async def test_an_override_no_tool_matches_applies_nothing_and_warns_once(publish: Publish) -> None:
    # The drift case: the tool was removed or renamed in code, or this deployment never had it. Tool
    # availability is dynamic across deployments and across steps, so by default it warns rather than
    # fails -- once per process, although the listing runs on every step.
    seen: list[ToolDefinition] = []
    publish(
        'unknown_tool',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='missing', description='ignored'),
                ToolDefinitionOverride(name='get_weather', description='Managed.'),
            ]
        ),
    )
    capability = AgentControl('unknown_tool', label='production')
    with pytest.warns(UserWarning, match=r"patches tool 'missing', which no toolset advertises") as caught:
        await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    assert len(caught) == 1
    assert advertised(seen) == {'get_weather': 'Managed.'}


async def test_an_override_narrowed_to_a_toolset_matches_only_that_toolsets_tool(publish: Publish) -> None:
    """`toolset` on an override is the same string the baseline reports, and it narrows the match.

    One config is applied across deployments, and two of them can each have a `search` that means
    something different. An entry narrowed to the CRM's `search` leaves the docs toolset's alone,
    and an entry narrowed to a toolset this deployment does not have reaches nothing -- even when a
    tool of that name exists somewhere else.
    """
    seen: list[ToolDefinition] = []

    def search(query: str) -> str:  # pragma: no cover - advertised only
        return query

    crm = FunctionToolset[object]([search], id='crm')
    docs = FunctionToolset[object]([get_forecast], id='docs')
    publish(
        'qualified',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='search', toolset='crm', description='Search CRM records.'),
                ToolDefinitionOverride(name='get_forecast', toolset='crm', description='never shown'),
            ]
        ),
    )
    capability = AgentControl('qualified', label='production')
    with pytest.warns(UserWarning, match=r"patches tool 'get_forecast' from toolset 'crm', which no toolset"):
        await Agent(capture_tools(seen), toolsets=[crm, docs], capabilities=[capability]).run('hello')
    assert advertised(seen) == {'search': 'Search CRM records.', 'get_forecast': None}


async def test_a_qualified_override_beats_an_unqualified_one_for_the_same_tool(publish: Publish) -> None:
    # Both entries are valid side by side: "every `get_weather`" and "the weather toolset's
    # `get_weather`". Where both match one tool the specific entry wins, and the general one is
    # outranked rather than unmatched -- it reached the tool it named -- so nothing warns.
    seen: list[ToolDefinition] = []
    publish(
        'qualified_precedence',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='get_weather', description='Any toolset.'),
                ToolDefinitionOverride(name='get_weather', toolset='weather', description='The weather toolset.'),
            ]
        ),
    )
    capability = AgentControl('qualified_precedence', label='production')
    with warnings.catch_warnings(record=True) as caught:
        await Agent(capture_tools(seen), toolsets=[weather_toolset()], capabilities=[capability]).run('hello')
    assert caught == []
    assert advertised(seen) == {'get_weather': 'The weather toolset.'}


async def test_two_overrides_with_the_same_toolset_and_name_keep_the_first(publish: Publish) -> None:
    seen: list[ToolDefinition] = []
    publish(
        'duplicate_qualified',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='get_weather', toolset='weather', description='First.'),
                ToolDefinitionOverride(name='get_weather', toolset='weather', description='Second.'),
            ]
        ),
    )
    capability = AgentControl('duplicate_qualified', label='production')
    with pytest.warns(UserWarning, match=r"names tool 'get_weather' from toolset 'weather' more than once"):
        await Agent(capture_tools(seen), toolsets=[weather_toolset()], capabilities=[capability]).run('hello')
    assert advertised(seen) == {'get_weather': 'First.'}


async def test_two_overrides_naming_the_same_tool_keep_the_first(publish: Publish) -> None:
    # The same first-wins rule as a duplicated instruction `id`, and the same reason: which entry a
    # colliding pair resolves to has to be a property of the config, not of JSON key order.
    seen: list[ToolDefinition] = []
    publish(
        'duplicate_tool',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(name='get_weather', description='First.'),
                ToolDefinitionOverride(name='get_weather', description='Second.'),
            ]
        ),
    )
    capability = AgentControl('duplicate_tool', label='production')
    with pytest.warns(UserWarning, match=r"names tool 'get_weather' more than once") as caught:
        await Agent(capture_tools(seen), tools=[get_weather], capabilities=[capability]).run('hello')
    # Read on every `get_tools`, warned about once.
    assert len(caught) == 1
    assert advertised(seen) == {'get_weather': 'First.'}


UNMATCHED_CASES = [
    pytest.param(
        {'instructions': [{'id': 'toolset:nope', 'instructions': 'never sent'}]},
        r"addresses instruction block 'toolset:nope', which this request does not assemble",
        id='an-instruction-id-no-block-carries',
    ),
    pytest.param(
        {'instructions': [{'id': 'toolset:weather', 'instructions': 'never sent'}]},
        r"addresses instruction block 'toolset:weather', which the agent recomputes per request",
        id='an-instruction-id-only-a-dynamic-block-carries',
    ),
    # An entry with no `instructions` is held back only when it would drop a block this request
    # assembles; one that reaches nothing is still the contract's to report, typo or not.
    pytest.param(
        {'instructions': [{'id': 'toolset:nope', 'text': 'never sent'}]},
        r"addresses instruction block 'toolset:nope', which this request does not assemble",
        id='an-instruction-id-no-block-carries-without-instructions',
    ),
    pytest.param(
        {'instructions': [{'id': 'toolset:weather', 'dynamic': True}]},
        r"addresses instruction block 'toolset:weather', which the agent recomputes per request",
        id='a-dynamic-block-as-a-baseline-lists-it',
    ),
    pytest.param(
        {'tool_definitions': [{'name': 'missing', 'description': 'never shown'}]},
        r"patches tool 'missing', which no toolset advertises",
        id='a-tool-override-no-tool-matches',
    ),
    pytest.param(
        {'tool_definitions': [{'name': 'get_weather', 'toolset': 'crm', 'description': 'never shown'}]},
        r"patches tool 'get_weather' from toolset 'crm', which no toolset advertises",
        id='a-tool-override-narrowed-to-a-toolset-this-agent-lacks',
    ),
    pytest.param(
        {'tool_definitions': [{'name': 'get_weather', 'parameters': {'missing': {'description': 'never shown'}}}]},
        r"patches parameter 'missing' of tool 'get_weather'",
        id='a-parameter-patch-the-tool-has-no-parameter-for',
    ),
    pytest.param(
        {
            'tool_definitions': [
                {'name': 'get_weather', 'new_name': 'get_forecast'},
                {'name': 'get_forecast', 'description': 'The other one.'},
            ]
        },
        r"renames 'get_weather' to 'get_forecast', which is already advertised",
        id='a-rename-another-advertised-tool-already-answers-to',
    ),
    pytest.param(
        {'settings': {'temperature': 0.2, 'service_tier': 'flex'}},
        r"sets 'service_tier', which this version of the Agent Control contract has no model setting for",
        id='a-settings-key-this-sdk-has-no-field-for',
    ),
    pytest.param(
        {'settings': {'timeout': -5}},
        r'sets a request timeout of -5.0 seconds, which is not a budget',
        id='a-timeout-that-is-not-a-request-budget',
    ),
    pytest.param(
        {'future_section': {'anything': 1}},
        r"publishes a 'future_section' section, which this version of the Agent Control contract",
        id='a-top-level-key-this-release-has-no-section-for',
    ),
    pytest.param(
        {
            'tool_definitions': [
                {'name': 'get_forecast', 'description': 'First.'},
                {'name': 'get_forecast', 'description': 'Second.'},
            ]
        },
        r"names tool 'get_forecast' more than once",
        id='the-same-tool-named-twice',
    ),
]


async def run_with_unmatched(
    capfire: CaptureLogfire, name: str, value: dict[str, Any], on_unmatched: Any
) -> list[InstructionPart]:
    """Run the block-of-each-kind agent plus the weather toolset's tool under one `on_unmatched` policy."""
    seen: list[InstructionPart] = []
    agent = Agent(
        capture_instructions(seen),
        instructions='AGENT: You are a concise checkout assistant.',
        tools=[get_forecast],
        toolsets=[weather_toolset()],
        capabilities=[AgentControl(name, label='production', on_unmatched=on_unmatched)],
    )
    with variables_provider(capfire, published_value(f'agent__{name}', value)):
        await agent.run('hello')
    return seen


@pytest.mark.parametrize('value,message', UNMATCHED_CASES)
async def test_on_unmatched_error_fails_the_run_with_the_warning_message(
    capfire: CaptureLogfire, value: dict[str, Any], message: str
) -> None:
    # For the deployment that would rather stop than run with part of its published config silently
    # unapplied. The message is the one `'warn'` would have emitted, so a warning someone tolerated
    # reads like the error they would have gotten by not tolerating it.
    with pytest.raises(UserError, match=message):
        await run_with_unmatched(capfire, 'unmatched_error', value, 'error')


@pytest.mark.parametrize('value,message', UNMATCHED_CASES)
async def test_on_unmatched_ignore_applies_nothing_and_says_nothing(
    capfire: CaptureLogfire, value: dict[str, Any], message: str
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        parts = await run_with_unmatched(capfire, 'unmatched_ignore', value, 'ignore')
    assert caught == []
    assert triples(parts) == [AGENT_BLOCK, TOOLSET_BLOCK]


@pytest.mark.parametrize('value,message', UNMATCHED_CASES)
async def test_on_unmatched_warn_is_the_default(capfire: CaptureLogfire, value: dict[str, Any], message: str) -> None:
    with pytest.warns(UserWarning, match=message):
        parts = await run_with_unmatched(capfire, 'unmatched_warn', value, 'warn')
    assert triples(parts) == [AGENT_BLOCK, TOOLSET_BLOCK]
    assert AgentControl('unmatched_default').on_unmatched == 'warn'


async def test_every_section_is_reported_together_for_the_whole_request(capfire: CaptureLogfire) -> None:
    """One report per request naming every section, rather than one report per section.

    `'error'` is the case that makes this load-bearing. The sections are applied by three different
    hooks -- the toolset listing, the settings contribution, the instruction hook -- so raising inside
    the first of them to run would fail the run naming the tool override and never plan the settings
    or the instructions, which would make the strictest policy the one that reports the least. Both
    policies are asserted on the same published value, because "the error says what the warning said"
    is the promise.
    """
    value = {
        'future_section': {'anything': 1},
        'instructions': [{'id': 'toolset:nope', 'instructions': 'never sent'}],
        'settings': {'service_tier': 'flex'},
        'tool_definitions': [{'name': 'missing', 'description': 'never shown'}],
    }
    fragments = [
        "publishes a 'future_section' section",
        "patches tool 'missing'",
        "sets 'service_tier'",
        "addresses instruction block 'toolset:nope'",
    ]

    with pytest.warns(UserWarning) as caught:
        await run_with_unmatched(capfire, 'every_section_warn', value, 'warn')
    warned = '\n'.join(str(warning.message) for warning in caught)
    assert len(caught) == len(fragments)
    assert [fragment for fragment in fragments if fragment not in warned] == []

    with pytest.raises(UserError) as error:
        await run_with_unmatched(capfire, 'every_section_error', value, 'error')
    assert [fragment for fragment in fragments if fragment not in str(error.value)] == []


async def test_schema_without_properties_is_tolerated(publish: Publish) -> None:
    seen: list[ToolDefinition] = []

    def raw_tool() -> str:  # pragma: no cover - advertised only
        return 'ok'

    tool = Tool.from_schema(raw_tool, name='raw_tool', description='Original.', json_schema={'type': 'object'})
    publish(
        'raw_schema',
        AgentConfig(
            tool_definitions=[
                ToolDefinitionOverride(
                    name='raw_tool', parameters={'missing': ParameterOverride(description='ignored')}
                )
            ]
        ),
    )
    capability = AgentControl('raw_schema', label='production')
    with pytest.warns(
        UserWarning, match=r"patches parameter 'missing' of tool 'raw_tool'.*has no top-level parameters"
    ):
        await Agent(capture_tools(seen), tools=[tool], capabilities=[capability]).run('hello')
    assert seen[0].parameters_json_schema == {'type': 'object'}


async def test_managed_model_runs_model_less_agent_and_run_model_wins(publish: Publish) -> None:
    publish('managed_model', AgentConfig(model='test'))
    managed = AgentControl('managed_model', label='production')
    assert (await Agent(None, capabilities=[managed]).run('hello')).output.startswith('success')

    def call_site(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('call-site')])

    result = await Agent(None, capabilities=[managed]).run('hello', model=FunctionModel(call_site))
    assert result.output == 'call-site'


async def test_unknown_managed_model_keeps_code_model(publish: Publish) -> None:
    # A model this deployment cannot infer is the drift case the `model` section has: the config names
    # a provider or model this release, or this install's extras, has no client for. The run keeps the
    # code model rather than failing, and says so once per process -- the config is resolved on every
    # run, so a warning per run would bury the signal under its own repetition.
    publish('unknown_model', AgentConfig(model='not-a-provider:not-a-model'))
    with pytest.warns(UserWarning, match='selects unknown model') as caught:
        agent = Agent(TestModel(), capabilities=[AgentControl('unknown_model', label='production')])
        result = await agent.run('hello')
        await agent.run('again')
    assert len(caught) == 1
    assert result.output.startswith('success')


async def test_nameless_model_selector_resolves_once_per_run(monkeypatch: pytest.MonkeyPatch, publish: Publish) -> None:
    # A nameless capability's selector is evaluated once per request step, but the managed model is a
    # run-stable value, so it memoizes and resolves the variable exactly once even across steps.
    resolves: list[str] = []
    original = AgentControl[Any]._resolve_for_selection  # pyright: ignore[reportPrivateUsage]

    def counting(self: AgentControl[Any], variable: Variable[Any], ctx: Any) -> Any:
        resolves.append(variable.name)
        return original(self, variable, ctx)

    monkeypatch.setattr(AgentControl, '_resolve_for_selection', counting)

    def a_tool() -> str:
        return 'ok'

    publish('multi_step', AgentConfig(model='test'))
    capability = AgentControl(label='production')
    # A model-less agent with one tool: `TestModel` calls the tool (step 1) then answers (step 2).
    await Agent(None, name='multi_step', tools=[a_tool], capabilities=[capability]).run('hello')
    assert resolves == ['agent__multi_step']


async def test_callable_targeting_resolution_is_reused_for_run(publish: Publish) -> None:
    calls = 0

    def targeting(_ctx: RunContext[object]) -> str:
        nonlocal calls
        calls += 1
        return f'key-{calls}'

    publish('callable_targeting', AgentConfig(model='test'))
    capability = AgentControl('callable_targeting', label='production', targeting_key=targeting)
    result = await Agent(TestModel(), capabilities=[capability]).run('hello')
    assert result.output.startswith('success')
    assert calls == 1


async def test_a_call_site_model_run_resolves_its_own_targeting(publish: Publish) -> None:
    """A run that skips model selection resolves for itself rather than reusing the previous run's.

    Model selection runs in the caller's task and `wrap_run` in a task started from a copy of its
    context, so a handoff `wrap_run` only reset on its own side stayed in the caller's context. The next
    run in the same task with `run(model=...)` skips selection and would then serve the previous run's
    resolution -- another tenant's instructions and tools -- without consulting its own targeting.
    """
    keys: list[str] = []

    def targeting(ctx: RunContext[str]) -> str:
        keys.append(ctx.deps)
        return ctx.deps

    publish('per_tenant', AgentConfig(model='test'))
    capability = AgentControl[str]('per_tenant', label='production', targeting_key=targeting)
    agent = Agent(TestModel(), deps_type=str, capabilities=[capability])
    await agent.run('hello', deps='tenant-a')
    await agent.run('hello', deps='tenant-b', model=TestModel())
    assert keys == ['tenant-a', 'tenant-b']


async def test_a_resolved_config_builds_one_snapshot_per_process(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The baseline walks every assembled instruction block and every advertised tool, and an agent
    # whose config is published reports it too -- so what keeps that off the request path is the
    # once-per-process guard, not the config having resolved. Two runs of an agent that calls a tool
    # is four model requests, and one snapshot.
    builds: list[AgentConfig] = []

    def counting(**kwargs: Any) -> AgentConfig:
        # The contract's own function, not the name patched onto the module under test.
        baseline = build_baseline(**kwargs)
        builds.append(baseline)
        return baseline

    monkeypatch.setattr(_agent_control, 'build_baseline', counting)
    with variables_provider(capfire, published_value('agent__known', {'model': 'test'})):
        agent = Agent(TestModel(), tools=[get_weather], capabilities=[AgentControl('known', label='production')])
        await agent.run('hello')
        await agent.run('again')
    assert len(builds) == 1


async def test_the_hooks_outside_a_run_are_inert() -> None:
    # Nothing is resolved until `wrap_run` opens the run's resolution context, so every hook that
    # reaches for it has to answer for a capability that was never entered -- as a graph built for
    # inspection is. Both halves of the split are asked, because each reaches for the same resolution.
    bound = AgentControl('outside_run').for_agent(Agent(TestModel()))
    assert isinstance(bound, CombinedCapability)
    request = cast(Any, object())
    for half in bound.capabilities:
        assert await half.before_model_request(cast(Any, None), request) is request
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    contributed = bound.capabilities[1].get_model_settings()
    assert callable(contributed) and contributed(ctx) == ModelSettings()


async def test_a_setting_the_baseline_leaves_out_is_still_sent(capfire: CaptureLogfire) -> None:
    """`extra_headers` and `extra_body` are left out of the baseline, not out of the request.

    Which keys a baseline may hold is the contract's rule and `test_config_hint.py` asserts it there.
    This is the other half of that promise: the run itself is unchanged by what the snapshot omits.
    """
    sent: list[ModelSettings | None] = []

    def model(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        sent.append(info.model_settings)
        return ModelResponse(parts=[TextPart('done')])

    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(
            FunctionModel(model),
            model_settings=cast(
                ModelSettings,
                {
                    'temperature': 0.1,
                    'extra_headers': {'Authorization': 'Bearer sk-secret'},
                    'openai_custom_option': object(),
                },
            ),
            capabilities=[AgentControl('secretless_baseline')],
        ).run('hello')

    assert sent[0] is not None and sent[0]['extra_headers'] == {'Authorization': 'Bearer sk-secret'}  # type: ignore[typeddict-item]
    assert sent[0]['openai_custom_option'] is not None  # type: ignore[typeddict-item]


async def test_rename_routes_to_the_tool_the_model_was_handed(capfire: CaptureLogfire) -> None:
    """A dynamic toolset may put a different tool behind an advertised name between listing and call.

    Routing reads the code-side name off the tool the model was handed, so the name every downstream
    consumer sees -- a name-based `ApprovalRequiredToolset` above all -- is the name of the tool that
    actually runs. Recomputing it from a fresh listing would authorize one tool and run the other.
    """
    listings = 0

    class Flipping(WrapperToolset[object]):
        """Advertises `first` on the first listing and `second` on every later one."""

        async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
            nonlocal listings
            tools = await super().get_tools(ctx)
            listings += 1
            return {('first' if listings == 1 else 'second'): tools['first' if listings == 1 else 'second']}

    authorized: list[str] = []

    class NamePolicy(WrapperToolset[object]):
        """Stands in for a name-based policy: it records the name it is asked to authorize."""

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
        ) -> Any:
            authorized.append(name)
            return await super().call_tool(name, tool_args, ctx, tool)

    def first() -> str:
        return 'ran first'

    def second() -> str:  # pragma: no cover - advertised on the second listing only
        return 'ran second'

    calls = 0

    def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('helper', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    managed = {'tool_definitions': [{'name': 'first', 'new_name': 'helper'}, {'name': 'second', 'new_name': 'helper'}]}
    with variables_provider(capfire, published_value('agent__rename_identity', managed)):
        await Agent(
            FunctionModel(model),
            toolsets=[NamePolicy(Flipping(FunctionToolset[object]([first, second])))],
            # Each listing advertises one of the two patched tools, so the other's override reaches
            # nothing on that listing: the dynamic case `on_unmatched` exists for, and not this test's.
            capabilities=[AgentControl('rename_identity', on_unmatched='ignore')],
        ).run('hello')

    assert authorized == ['first']
