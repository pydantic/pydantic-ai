"""The span an agent reports its code baseline on.

The SDK never creates or updates a managed variable. Every agent emits one
`agent_control_config_hint` span per process carrying everything a config would be created from, and
creating one -- or refreshing a stored baseline the code has moved on from -- is a Logfire-side flow.
These tests are therefore the contract the platform side consumes: the span's name, its attributes,
when it is and is not emitted, and how a baseline too large for a span attribute degrades.

That nothing writes is asserted for every test in this package by the `_refuse_variable_writes`
fixture in `conftest.py`; `test_a_run_writes_nothing_to_the_variable_api` says it once explicitly.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import logfire
import pytest
from logfire.agent_control import SCHEMA_SHA256, canonical_json
from logfire.testing import CaptureLogfire
from logfire.variables import Rollout, VariableConfig, VariablesConfig
from logfire.variables.local import LocalVariableProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.logfire import AgentControl, _agent_control

from ._helpers import Publish, get_weather, variables_provider, weather_toolset

_SPAN_NAME = 'agent_control_config_hint'


def hints(capfire: CaptureLogfire) -> list[dict[str, Any]]:
    """The attributes of each config-hint span the run exported, in order."""
    return [span['attributes'] for span in capfire.exporter.exported_spans_as_dict() if span['name'] == _SPAN_NAME]


def baseline(attributes: dict[str, Any]) -> Any:
    """The `AgentConfig` a hint carries, parsed."""
    return json.loads(attributes['agent_control.baseline'])


def existing_variable(name: str) -> VariablesConfig:
    """A project that knows `name` but has no value published under any label."""
    return VariablesConfig(
        variables={name: VariableConfig(name=name, labels={}, rollout=Rollout(labels={}), overrides=[])}
    )


async def test_an_unconfigured_agent_reports_its_whole_baseline(capfire: CaptureLogfire) -> None:
    def lookup(city: str) -> str:
        """Look up a city.

        Args:
            city: City to look up.
        """
        return city

    def raw() -> str:
        return 'raw'

    raw_tool = Tool.from_schema(
        raw,
        name='raw',
        description=None,
        json_schema={
            'type': 'object',
            'properties': {'plain': 'not-a-schema', 'count': {'description': 5}, 'named': {'description': 'Named.'}},
        },
    )
    empty_tool = Tool.from_schema(raw, name='empty', description=None, json_schema={'type': 'object'})

    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(
            TestModel(),
            name='Snapshot Agent',
            instructions='Code instructions.',
            model_settings={'temperature': 0.3},
            tools=[lookup],
            # One toolset with an `id` and one without: the baseline reports the id when there is
            # one and falls back to the label, so the UI can group tools by origin either way.
            toolsets=[FunctionToolset([raw_tool], id='raw-tools'), FunctionToolset([empty_tool])],
            capabilities=[AgentControl()],
        )
        await agent.run('hello')

    assert len(hints(capfire)) == 1
    attributes = hints(capfire)[0]
    # The identity half of the contract. `agent_name` is the name as written and `variable_name` is
    # what normalizing it produced, which is how a consumer tells two agents apart after they have
    # landed on one key.
    assert attributes['agent_control.variable_name'] == 'agent__snapshot_agent'
    assert attributes['agent_control.agent_name'] == 'Snapshot Agent'
    assert attributes['agent_control.framework'] == 'pydantic-ai'
    assert attributes['agent_control.baseline_source'] == 'code'
    assert attributes['agent_control.schema_sha256'] == SCHEMA_SHA256
    assert attributes['agent_control.baseline_reduction'] == 'none'
    assert attributes['agent_control.baseline_bytes'] == len(attributes['agent_control.baseline'].encode())
    # Every agent reports, so the span's existence no longer says whether this one had a config. The
    # reason is what says it: `'code_default'` is a baseline waiting for a config to be created from
    # it, where `'resolved'` is one that may only be refreshing a stale `example`.
    assert attributes['agent_control.resolution_reason'] == 'code_default'
    # The message names no agent, so it stays one string across a project's agents. The span name is
    # separate from it, so the message can be reworded without moving what a query selects on.
    assert attributes['logfire.msg'] == 'Agent Control reported the code baseline for this agent'

    assert baseline(attributes) == {
        'instructions': [{'id': 'agent', 'instructions': 'Code instructions.', 'dynamic': False}],
        'model': 'test:test',
        'settings': {'temperature': 0.3},
        'tool_definitions': [
            {
                'name': 'lookup',
                'description': 'Look up a city.',
                'parameters': {'city': {'description': 'City to look up.'}},
                'toolset': '<agent>',
            },
            # An undocumented parameter is listed with nothing in it, which is the point: it is
            # exactly the one somebody wants to describe from Logfire, and a baseline that listed
            # only the documented ones would hide it until it had been documented in code first.
            {
                'name': 'raw',
                'parameters': {'plain': {}, 'count': {}, 'named': {'description': 'Named.'}},
                'toolset': 'raw-tools',
            },
            {'name': 'empty', 'toolset': 'FunctionToolset'},
        ],
    }


async def test_the_hint_says_which_deployment_reported_the_baseline(capfire: CaptureLogfire) -> None:
    """A variable is derived from the agent's name alone, so the span has to say whose code this is.

    Two services that each define a `checkout_assistant`, and the same service's dev and prod
    deployments, all land on one `agent__checkout_assistant`: a variable is one value per project and
    the dev/prod split is its labels. The identity comes off the Logfire instance the hint is emitted
    on, so it is whatever the deployment already told `logfire.configure()` rather than something to
    configure twice.
    """
    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        service_name='checkout',
        service_version='1a2b3c4',
        environment='prod',
        variables=logfire.LocalVariablesOptions(config=VariablesConfig(variables={})),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    await Agent(TestModel(), name='identified', capabilities=[AgentControl(logfire_instance=instance)]).run('hello')

    attributes = hints(capfire)[0]
    assert attributes['agent_control.service_name'] == 'checkout'
    assert attributes['agent_control.service_version'] == '1a2b3c4'
    assert attributes['agent_control.environment'] == 'prod'


async def test_the_hint_is_reported_as_written_with_scrubbing_at_its_default(capfire: CaptureLogfire) -> None:
    """The baseline reaches Logfire as the code wrote it, with scrubbing left at its default.

    Logfire's scrubbing matches substrings, and `auth`, `session` and `token` are ordinary words in a
    prompt, a tool description, an agent's name and a service's name. Every string here matches one.
    The baseline is the document the UI promotes into the config's `example`, so a redaction inside it
    is a corrupted document rather than a hidden secret -- one that no longer matches
    `agent_control.baseline_sha256` or `agent_control.baseline_bytes`, both taken over the baseline
    before it is exported, and one that would put `[Scrubbed due to 'auth']` in front of the model if
    somebody took that block over by id. A redacted `agent_control.variable_name` is worse still: the
    hint names no variable, so the agent never appears in Logfire and nothing is raised anywhere.

    The exemption is `BaseScrubber.SAFE_KEYS` in the `logfire` package, which is where the six
    attribute names live; this is the adapter's half of that contract.
    """

    def refund_order(order_id: str) -> str:
        """Refund an order the customer has authorization for.

        Args:
            order_id: The order to refund.
        """
        return order_id

    instance = logfire.configure(
        local=True,
        send_to_logfire=False,
        console=False,
        # A service that serves checkout sessions, a preview environment named after the branch it
        # was built from, and a version string carrying that branch name.
        service_name='checkout-session-api',
        service_version='1.4.0+authz.2',
        environment='pr-auth-refresh',
        variables=logfire.LocalVariablesOptions(config=VariablesConfig(variables={})),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    instructions = 'Order tools are authoritative for status and refunds.'
    await Agent(
        TestModel(),
        name='auth_router',
        instructions=instructions,
        tools=[refund_order],
        capabilities=[AgentControl(logfire_instance=instance)],
    ).run('hello')

    attributes = hints(capfire)[0]
    carried = baseline(attributes)
    assert carried['instructions'] == [{'id': 'agent', 'instructions': instructions, 'dynamic': False}]
    assert carried['tool_definitions'][0]['description'] == 'Refund an order the customer has authorization for.'
    assert attributes['agent_control.variable_name'] == 'agent__auth_router'
    assert attributes['agent_control.agent_name'] == 'auth_router'
    assert attributes['agent_control.service_name'] == 'checkout-session-api'
    assert attributes['agent_control.service_version'] == '1.4.0+authz.2'
    assert attributes['agent_control.environment'] == 'pr-auth-refresh'
    # The two promises a reduction of `'none'` makes to a consumer, checked against the document the
    # span carries rather than the one this process built.
    assert attributes['agent_control.baseline_reduction'] == 'none'
    assert attributes['agent_control.baseline_sha256'] == hashlib.sha256(canonical_json(carried)).hexdigest()
    assert attributes['agent_control.baseline_bytes'] == len(attributes['agent_control.baseline'].encode())
    # Scrubbing records what it rewrote, so its absence covers every attribute of the span rather
    # than the ones named above -- including the run's own `logfire.variables.agent__auth_router`,
    # whose key is built from the variable's name and is covered by a safe key *prefix*.
    assert 'logfire.scrubbed' not in attributes


async def test_identity_the_sdk_does_not_know_is_left_off(capfire: CaptureLogfire) -> None:
    # Absent rather than `''`: absent is a state a consumer can act on -- group these hints by
    # deployment, or say it cannot -- where an empty string is a value it has to learn to disbelieve.
    # `service_version` is not asserted here: Logfire fills it in from the commit of the checkout the
    # process runs in, so what it holds depends on where the tests are run rather than on this code.
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(TestModel(), name='anonymous_deployment', capabilities=[AgentControl()]).run('hello')

    attributes = hints(capfire)[0]
    assert 'agent_control.service_name' not in attributes
    assert 'agent_control.environment' not in attributes


async def test_the_baseline_digest_changes_only_when_the_code_does(capfire: CaptureLogfire) -> None:
    # The guard reports one baseline per process per variable, so a code change that happens while the
    # process runs is never re-reported. The digest is what makes that detectable at all: a consumer
    # holding an earlier hint can tell "the same baseline again" from "this agent has moved".
    with variables_provider(capfire, VariablesConfig(variables={})):
        for name, instructions in (
            ('digest_one', 'CODE one.'),
            ('digest_two', 'CODE two.'),
            ('digest_one_again', 'CODE one.'),
        ):
            await Agent(TestModel(), name=name, instructions=instructions, capabilities=[AgentControl()]).run('hello')

    one, two, again = (attributes['agent_control.baseline_sha256'] for attributes in hints(capfire))
    assert one != two
    assert one == again


def test_the_baseline_digest_uses_the_contracts_canonical_json() -> None:
    """One definition of canonical, and it is the contract's.

    This used to pin a copy of the canonical form kept here against the contract's, because the
    contract's helper was private. It is exported now, so there is one function and nothing to drift
    -- what is left worth asserting is the form itself, since a digest is only comparable across
    languages while all three of its properties hold: sorted keys, `(',', ':')` separators, and
    `ensure_ascii=False`. The probe is non-ASCII on purpose: `ensure_ascii` is the flag an ASCII
    document cannot tell apart, and the one a prompt in any other language would expose first
    (`json.dumps` escapes it by default and `JSON.stringify` does not).
    """
    document = {'model': 'test:test', 'instructions': 'Grüße, ¿cómo estás?'}
    # All three properties, readable in one line: `instructions` sorts before `model`, the separators
    # carry no spaces, and the accents survive unescaped.
    assert canonical_json(document).decode() == '{"instructions":"Grüße, ¿cómo estás?","model":"test:test"}'


async def test_the_hint_lists_every_instruction_block(capfire: CaptureLogfire) -> None:
    # The baseline the Logfire UI diffs managed values against, and the reason it can offer an
    # override per block rather than one copy-the-whole-prompt button: the joined prompt telemetry
    # records has no seams, so a snapshot taken from that could only be copied wholesale -- which,
    # since managed instructions *add*, would send the agent's own text twice with a frozen date in
    # the middle.
    agent = Agent(
        TestModel(),
        name='blocks_snapshot',
        instructions='AGENT: You are a concise checkout assistant.',
        toolsets=[weather_toolset()],
        capabilities=[AgentControl()],
    )

    @agent.instructions(name='today')
    def today(_ctx: RunContext[object]) -> str:
        return 'DYNAMIC: today is Monday.'

    @agent.instructions
    def unnamed(_ctx: RunContext[object]) -> str:
        return 'UNNAMED: no declared id.'

    with variables_provider(capfire, VariablesConfig(variables={})):
        await agent.run('hello')

    attributes = hints(capfire)[0]
    assert baseline(attributes)['instructions'] == [
        {'id': 'agent', 'instructions': 'AGENT: You are a concise checkout assistant.', 'dynamic': False},
        # A dynamic block contributes its seam and not its text: what it rendered to here is one
        # request's answer, built from whatever that run carried. The `id` and the flag are what the
        # editor needs -- enough to show the block and that it is recomputed per request.
        {'id': 'agent:today', 'dynamic': True},
        {'id': 'toolset:weather', 'dynamic': True},
    ]
    # `UNNAMED: no declared id.` is absent entirely: the function declared no id, so there is nothing
    # to address it by, and it is dynamic, so there is no text to report. Nothing left to say about it.
    assert 'UNNAMED' not in attributes['agent_control.baseline']


async def test_a_dynamic_blocks_rendered_text_never_reaches_the_hint(capfire: CaptureLogfire) -> None:
    """An instruction function reads the run: a tenant, a user, a retrieved document.

    A hint is exported to the Logfire project, so it describes the agent rather than recording a
    request. A dynamic block therefore contributes its `id` and its flag and nothing else -- which is
    also all an editor needs to show it and to not offer to change it.
    """
    agent = Agent(
        TestModel(),
        name='no_request_data',
        deps_type=str,
        instructions='You are a support agent.',
        capabilities=[AgentControl()],
    )

    @agent.instructions(name='tenant')
    def tenant(ctx: RunContext[str]) -> str:
        return f'You are serving tenant {ctx.deps}. Their account token is tok_SECRET_9f2.'

    with variables_provider(capfire, VariablesConfig(variables={})):
        await agent.run('hello', deps='ACME Health (patient records)')

    attributes = hints(capfire)[0]
    assert 'tok_SECRET_9f2' not in attributes['agent_control.baseline']
    assert 'ACME Health' not in attributes['agent_control.baseline']
    assert baseline(attributes)['instructions'] == [
        {'id': 'agent', 'instructions': 'You are a support agent.', 'dynamic': False},
        {'id': 'agent:tenant', 'dynamic': True},
    ]


async def test_only_canonical_settings_reach_the_hint(capfire: CaptureLogfire) -> None:
    """`extra_headers` and `extra_body` are forwarded to the provider and routinely carry authorization.

    The baseline holds the canonical keys only -- because that is all `AgentConfigSettings` has fields
    for, not because of a list of names to withhold. The run itself still sends everything.
    """
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(
            TestModel(),
            name='secretless_hint',
            model_settings={
                'temperature': 0.1,
                'extra_headers': {'Authorization': 'Bearer sk-secret'},
                'extra_body': {'signature': 'sk-secret'},
            },
            capabilities=[AgentControl()],
        ).run('hello')

    attributes = hints(capfire)[0]
    assert baseline(attributes)['settings'] == {'temperature': 0.1}
    assert 'sk-secret' not in attributes['agent_control.baseline']


async def test_the_baseline_is_the_code_and_not_what_a_managed_config_would_do(capfire: CaptureLogfire) -> None:
    # A config published under a label this capability does not select leaves the run on the
    # code-defined agent, so the hint still fires -- and what it carries has to be the code, not the
    # value that was not applied.
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(
            TestModel(),
            name='code_only_hint',
            instructions='CODE instruction.',
            model_settings={'temperature': 0.1},
            tools=[get_weather],
            capabilities=[AgentControl(label='production')],
        ).run('hello')

    assert baseline(hints(capfire)[0]) == {
        'instructions': [{'id': 'agent', 'instructions': 'CODE instruction.', 'dynamic': False}],
        'model': 'test:test',
        'settings': {'temperature': 0.1},
        'tool_definitions': [{'name': 'get_weather', 'parameters': {'city': {}}, 'toolset': '<agent>'}],
    }


async def test_a_configured_agent_still_reports_its_code_baseline(capfire: CaptureLogfire, publish: Publish) -> None:
    """A config reaching the run does not stop the agent describing itself.

    The baseline the Logfire editor diffs against is a copy of the code, and code moves: an agent that
    reported only while unconfigured would go quiet the moment somebody configured it, and its stored
    baseline would describe the deployment it was created from forever. So the hint says what the code
    says now, whatever is published, and the reason says which of the two jobs it is for. What it
    carries is still the code and never the managed value -- that is what makes a diff a diff.
    """
    publish('configured', {'instructions': 'MANAGED: be brief.'})
    await Agent(
        TestModel(), name='configured', instructions='code', capabilities=[AgentControl(label='production')]
    ).run('hello')

    attributes = hints(capfire)[0]
    assert attributes['agent_control.resolution_reason'] == 'resolved'
    assert baseline(attributes)['instructions'] == [{'id': 'agent', 'instructions': 'code', 'dynamic': False}]


async def test_an_agent_reports_once_per_process(capfire: CaptureLogfire) -> None:
    # The provider is left knowing nothing, so the second run resolves nothing either: the guard is
    # what keeps it from reporting again, not a config that now exists.
    with variables_provider(capfire, VariablesConfig(variables={})):
        agent = Agent(TestModel(), name='reported_once', capabilities=[AgentControl()])
        await agent.run('hello')
        await agent.run('hello again')

    assert len(hints(capfire)) == 1


async def test_a_configured_agent_reports_once_per_process_too(capfire: CaptureLogfire, publish: Publish) -> None:
    # The guard is what bounds the cost of reporting unconditionally: the baseline walks every
    # assembled block and every advertised tool, and an agent whose config is published pays that on
    # its first request in the process and on no other.
    publish('configured_once', {'instructions': 'MANAGED: be brief.'})
    agent = Agent(
        TestModel(), name='configured_once', instructions='code', capabilities=[AgentControl(label='production')]
    )
    await agent.run('hello')
    await agent.run('hello again')

    assert len(hints(capfire)) == 1


async def test_rebuilding_the_agent_reports_once_per_process_too(capfire: CaptureLogfire) -> None:
    """Two `Agent` objects for one config in one process report once between them, not once each.

    The tests above reuse one agent across its runs, so they pass whether the guard is per process or
    merely per capability. This is the case that separates them, and it is the common one: an agent
    built inside a request handler is a new `Agent`, a new `AgentControl`, and a new `Variable` on
    every request.

    It caught a real defect. The guard was keyed on `variable.logfire_instance`, and
    `Variable.__init__` stores `logfire_instance.with_settings(...)` -- a new `Logfire` per variable,
    with no value equality -- so no two variables ever shared a key and the guard deduplicated
    nothing beyond repeat runs of one long-lived agent. Per-request construction reported a hint,
    with a full baseline walk, on every request.
    """
    with variables_provider(capfire, VariablesConfig(variables={})):
        for _ in range(2):
            agent = Agent(TestModel(), name='rebuilt_each_time', instructions='code', capabilities=[AgentControl()])
            await agent.run('hello')

    assert len(hints(capfire)) == 1


async def test_normalized_name_collision_reports_each_code_identity(capfire: CaptureLogfire) -> None:
    """Two written names sharing one variable each report the baseline that may differ in code."""
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(TestModel(), name='checkout-assistant', instructions='FIRST', capabilities=[AgentControl()]).run(
            'hello'
        )
        await Agent(TestModel(), name='checkout_assistant', instructions='SECOND', capabilities=[AgentControl()]).run(
            'hello'
        )

    attributes = hints(capfire)
    assert [hint['agent_control.variable_name'] for hint in attributes] == [
        'agent__checkout_assistant',
        'agent__checkout_assistant',
    ]
    assert [hint['agent_control.agent_name'] for hint in attributes] == [
        'checkout-assistant',
        'checkout_assistant',
    ]
    assert [baseline(hint)['instructions'][0]['instructions'] for hint in attributes] == ['FIRST', 'SECOND']


async def test_each_logfire_project_is_reported_to(capfire: CaptureLogfire) -> None:
    # A process can serve several Logfire projects, and a config in the first is not a config in the
    # second, so the guard is keyed by destination as well as by variable name. Each hint also has to
    # be emitted on its own project's instance rather than on whichever one happens to be the default.
    instances = [
        logfire.configure(
            local=True,
            send_to_logfire=False,
            console=False,
            variables=logfire.LocalVariablesOptions(config=VariablesConfig(variables={})),
            additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
        )
        for _ in range(2)
    ]

    for instance in instances:
        agent = Agent(TestModel(), name='two_projects', capabilities=[AgentControl(logfire_instance=instance)])
        await agent.run('hello')
        # A second run against the same instance is still guarded.
        await agent.run('hello again')

    assert [attributes['agent_control.variable_name'] for attributes in hints(capfire)] == [
        'agent__two_projects',
        'agent__two_projects',
    ]


async def test_a_variable_with_no_published_value_is_still_reported(capfire: CaptureLogfire) -> None:
    """A variable that exists but holds no value leaves the run on the code-defined agent, so it reports.

    The SDK cannot tell that case from an unknown variable: logfire collapses "no entry for this
    name", "no targeted value", and "no provider at all" into one `'code_default'` reason, and the
    only way to narrow it was a pre-flight call to the variable-management API on the run's thread --
    a blocking round trip whose best-effort failure suppressed the signal. Deduplicating a hint
    against the variables that already exist is the platform's job, and the platform is the side that
    actually knows.
    """
    with variables_provider(capfire, existing_variable('agent__known_but_empty')):
        await Agent(TestModel(), name='known_but_empty', capabilities=[AgentControl()]).run('hello')

    assert [attributes['agent_control.variable_name'] for attributes in hints(capfire)] == ['agent__known_but_empty']


async def test_an_agent_with_no_variables_provider_is_still_reported(capfire: CaptureLogfire) -> None:
    # Registration used to need the variable-management API, which a process holding only a
    # span-write token does not have. A hint travels the span pipeline, so that process registers too.
    await Agent(TestModel(), name='no_provider_hint', capabilities=[AgentControl()]).run('hello')

    assert [attributes['agent_control.variable_name'] for attributes in hints(capfire)] == ['agent__no_provider_hint']


async def test_the_hint_survives_a_raised_min_level(capfire: CaptureLogfire) -> None:
    """A hint is a span and not a log record, so a logging threshold cannot withhold it.

    `min_level` drops a log below it before it is ever exported, and a span with no level of its own
    is not subject to it. The platform side of Agent Control depends on this signal arriving, which
    is not something a project's logging configuration should get a vote on.
    """
    logfire.configure(
        send_to_logfire=False,
        console=False,
        min_level='warn',
        variables=logfire.LocalVariablesOptions(config=VariablesConfig(variables={})),
        additional_span_processors=[SimpleSpanProcessor(capfire.exporter)],
    )
    try:
        await Agent(TestModel(), name='quiet_project', capabilities=[AgentControl()]).run('hello')
    finally:
        logfire.configure(send_to_logfire=False, console=False)

    assert [attributes['agent_control.variable_name'] for attributes in hints(capfire)] == ['agent__quiet_project']


async def test_an_explicitly_named_capability_on_a_nameless_agent_reports_no_agent_name(
    capfire: CaptureLogfire,
) -> None:
    # An explicit capability `name` decouples the config from the agent's own name, and Pydantic AI
    # only infers an agent name from the variable an agent was assigned to -- which this one is not.
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(TestModel(), capabilities=[AgentControl('detached')]).run('hello')

    attributes = hints(capfire)[0]
    assert attributes['agent_control.variable_name'] == 'agent__detached'
    # Left off the span rather than carried as a null: absent says "there is none", where a
    # null-valued attribute would only raise the question of whether that is what the agent is called.
    assert 'agent_control.agent_name' not in attributes


async def test_an_agent_that_never_reaches_a_model_reports_nothing(capfire: CaptureLogfire) -> None:
    # The baseline is read off an assembled request, so there is nothing to report before one exists.
    # Constructing the capability and binding it to an agent must not report on their own.
    with variables_provider(capfire, VariablesConfig(variables={})):
        AgentControl().for_agent(Agent(TestModel(), name='never_run'))

    assert hints(capfire) == []


async def test_an_oversized_baseline_drops_its_tool_definitions(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A baseline over the budget gives up whole sections rather than being cut to length.

    The backend enforces its own attribute budget by truncating a long string in place, which for
    JSON yields an attribute that still looks like a string and no longer parses. So the reduction
    happens here, is named on the span, and leaves a whole valid `AgentConfig` behind.
    """
    # A budget between this agent's full baseline and the same baseline without its tool definitions,
    # so the first rung of the ladder is the one taken. Tightening it real-world is what a 64 KiB
    # instruction block and a few dozen tool schemas do on their own.
    monkeypatch.setattr(_agent_control, '_MAX_BASELINE_BYTES', 200)
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(
            TestModel(),
            name='oversized_tools',
            instructions='CODE instruction.',
            tools=[get_weather],
            capabilities=[AgentControl()],
        ).run('hello')

    attributes = hints(capfire)[0]
    assert attributes['agent_control.baseline_reduction'] == 'tool_definitions'
    # Still a parseable, whole config, with the section that carries the unbounded part left out.
    assert baseline(attributes) == {
        'instructions': [{'id': 'agent', 'instructions': 'CODE instruction.', 'dynamic': False}],
        'model': 'test:test',
    }
    # The size reported is the full baseline's, so a consumer sees how far over the budget it was
    # rather than how big the part that survived is.
    assert attributes['agent_control.baseline_bytes'] > len(attributes['agent_control.baseline'].encode())


async def test_a_baseline_too_large_even_reduced_is_omitted_and_says_so(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing is guessed at and nothing is cut: the hint still registers the agent, and the reduction
    # says why no baseline is on it.
    monkeypatch.setattr(_agent_control, '_MAX_BASELINE_BYTES', 10)
    with variables_provider(capfire, VariablesConfig(variables={})):
        await Agent(
            TestModel(),
            name='omitted_baseline',
            instructions='CODE instruction.',
            tools=[get_weather],
            capabilities=[AgentControl()],
        ).run('hello')

    attributes = hints(capfire)[0]
    assert attributes['agent_control.baseline_reduction'] == 'omitted'
    assert 'agent_control.baseline' not in attributes
    assert attributes['agent_control.variable_name'] == 'agent__omitted_baseline'
    assert attributes['agent_control.baseline_bytes'] > 10


async def test_a_reduced_baseline_still_digests_the_whole_one(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reduction changes what the span carries, and must not change what the digest says.

    The same code reported twice, once under a budget that leaves no room for a baseline at all. A
    digest taken after the reduction would make those two look like different agents -- and would
    leave every omitted report looking like every other one, which is the case with nothing else on
    the span to tell it apart by.
    """

    def agent(name: str) -> Agent[None, str]:
        return Agent(
            TestModel(),
            name=name,
            instructions='CODE instruction.',
            tools=[get_weather],
            capabilities=[AgentControl()],
        )

    with variables_provider(capfire, VariablesConfig(variables={})):
        await agent('whole_baseline').run('hello')
        with monkeypatch.context() as clamped:
            clamped.setattr(_agent_control, '_MAX_BASELINE_BYTES', 10)
            await agent('clamped_baseline').run('hello')

    whole, clamped_attributes = hints(capfire)
    assert whole['agent_control.baseline_reduction'] == 'none'
    assert clamped_attributes['agent_control.baseline_reduction'] == 'omitted'
    assert clamped_attributes['agent_control.baseline_sha256'] == whole['agent_control.baseline_sha256']


def test_the_budget_stays_an_order_of_magnitude_under_the_backends() -> None:
    # A guard on the constant itself. The whole point of enforcing a budget here is to stay well
    # under the row budget the backend truncates against, so a change to it is a decision to make
    # deliberately rather than a number to drift.
    assert _agent_control._MAX_BASELINE_BYTES == 1024 * 1024  # pyright: ignore[reportPrivateUsage]


async def test_a_run_writes_nothing_to_the_variable_api(capfire: CaptureLogfire) -> None:
    """The removed halves of this capability, asserted as the absence they now are.

    Creating the variable from the code baseline, and read-modify-writing its `example` to keep that
    baseline current, both went through these methods. Recorded rather than refused here, even though
    the package-wide `_refuse_variable_writes` fixture already refuses them, so that this test says
    what it is about instead of leaving a fixture to fail on its behalf.
    """
    calls: list[str] = []

    def record(method: str) -> Any:
        def recorded(_self: LocalVariableProvider, *_args: Any, **_kwargs: Any) -> None:
            calls.append(method)

        return recorded

    with pytest.MonkeyPatch.context() as monkeypatch:
        for method in ('create_variable', 'update_variable', 'delete_variable'):
            monkeypatch.setattr(LocalVariableProvider, method, record(method))
        with variables_provider(capfire, VariablesConfig(variables={})):
            agent = Agent(TestModel(), name='no_writes', instructions='code', capabilities=[AgentControl()])
            result = await agent.run('hello')
            await agent.run('again')
            assert calls == []
            # And the recorder is live, so an empty list means the runs called nothing rather than
            # that the patch landed somewhere the capability never reaches.
            provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
            provider.create_variable(existing_variable('agent__probe').variables['agent__probe'])

    assert result.output.startswith('success')
    assert calls == ['create_variable']
    assert len(hints(capfire)) == 1


async def test_the_packages_no_write_guard_refuses_a_write(capfire: CaptureLogfire) -> None:
    """`conftest.py`'s `_refuse_variable_writes` is what makes every test here an assertion about writing.

    Exercised directly so that "nothing in this package writes" is a property something enforces
    rather than a fixture no test ever reached.
    """
    with variables_provider(capfire, VariablesConfig(variables={})):
        provider = logfire.DEFAULT_LOGFIRE_INSTANCE.config.get_variable_provider()
        config = existing_variable('agent__guarded').variables['agent__guarded']
        with pytest.raises(AssertionError, match="called 'update_variable' on the variable provider"):
            provider.update_variable('agent__guarded', config)
