"""Live Agent Control conformance tests against a running Logfire platform.

The unit suite in `tests/logfire_variables` drives `AgentControl` through a local variable provider
and test models. It answers whether the capability applies a config correctly. What it cannot answer
is whether the *feature* works, and that is what this file is for: a test that needs a published
config puts one in the project through the platform's own variables API, every test resolves through
the real Logfire SDK and makes a real model request, and each assertion reads what actually
happened -- the request the model was handed, the spans the process exported, or the platform's own
copy of one of them.

Eighteen claims, one or more tests each:

1. An unconfigured agent runs exactly as written, and reports that nothing resolved.
2. One `agent_control_config_hint` span per process carries every attribute it promises, and
   arrives at the platform.
3. The baseline describes the code, including the dynamic block's absent text.
4. The config can be created through the API the way the promote-a-hint flow will have to.
5. An instruction block can be replaced by `id`, whole or named, and nothing else moves.
6. An added block lands after the last static block.
7. A block can be dropped, and removing the config puts every code block back.
8. A renamed tool reaches the model under the managed name and the code under its own.
9. Tool and parameter descriptions reach the model; the parameter schema stays code-owned.
10. A published model overrides the code model, across providers.
11. Published settings patch over the code settings rather than replacing them.
12. A call-site `run(model=...)` beats the published model, which beats `Agent(model=...)`.
13. `on_unmatched` warns once per process, or fails the run naming every issue at once, or is silent.
14. A config section this release has never heard of is reported by name, and its siblings apply.
15. Neither an unparsable value nor an unreachable platform takes a run down.
16. Two labels on one variable, and each run reports the label and version it resolved.
17. A config is picked up once per run, so publishing mid-run takes effect on the next one.
18. A configured agent still reports its code baseline, marked `resolved`.

**This suite creates, publishes over and deletes variables on the project it is pointed at**, before
and after every test, because a conformance suite for a feature whose whole surface is stored state
has to own the state it checks. Every name it touches begins with the per-run
`agent__harness_agent_control_live_<32 hex>`, so the only config it can delete is one this run
created -- and it refuses to run at all without `LOGFIRE_PLATFORM_ALLOW_WRITES=1`. See `README.md`
for the rest, including how to bring a platform up and the two traps worth knowing about.

Unlike the three suites beside it, no CI job runs this one: those need one container each, and this
needs a whole Logfire platform. Run it by hand with `make integration-logfire-platform`.

The demo this was ported from ran every scenario in its own process, because three of the things
being checked are once *per process*: the hint span, an `on_unmatched` warning, and the resolved
config. Here `fresh_process_state` clears the first two guards between tests instead (the same two
`tests/logfire_variables/conftest.py` clears), and every test builds a new agent for the third.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from collections.abc import Iterator

import logfire
import pytest
from logfire.agent_control import (
    AGENT_CONFIG_JSON_SCHEMA,
    SCHEMA_SHA256,
    AgentConfig,
    InstructionBlock,
    agent_variable_name,
    canonical_json,
)
from logfire.agent_control._reporting import reset_warned_messages

from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ToolCallPart

# Private on purpose, and imported the same way `tests/logfire_variables/conftest.py` imports it: the
# two once-per-process guards have no public reset, and a suite whose outcome must not depend on the
# order its tests ran in has to reach both.
from pydantic_ai_harness.logfire import _agent_control

from ._agent import (
    CANARY_LABEL,
    CODE_BLOCKS,
    CODE_MODEL,
    DYNAMIC_BLOCK_ID,
    DYNAMIC_DEPS_TOKENS,
    MANAGED_MODEL,
    PRODUCTION_LABEL,
    LiveAgent,
    ToolCall,
    build_agent,
    todays_date,
)
from ._observability import (
    ENVIRONMENT,
    HINT_SPAN,
    SERVICE_NAME,
    SERVICE_VERSION,
    SpanCapture,
    configure,
    configured_against,
    flush,
    refresh_variables,
)
from ._platform import (
    AGENT_NAME,
    VARIABLE_NAME,
    Platform,
    platform_target,
    require_provider_key,
    require_span_read_back,
)

CODE_BLOCK_ORDER = [
    'agent',
    'agent:escalation',
    'capability:store_policy',
    'toolset:orders',
    'toolset:catalog',
    DYNAMIC_BLOCK_ID,
]
"""The prompt as the code assembles it: the static blocks in source order, the dynamic one last."""

CODE_TOOL_ORDER = ['lookup_order', 'refund_order', 'search_catalog', 'lookup_stock']

FLAMINGO = 'You are the FLAMINGO storefront support agent. Begin every reply with the exact token <<FLAMINGO>>.'
CANARY = 'You are the CANARY build. Begin every reply with the exact token <<CANARY>>.'
ADDED_BLOCK = 'House rule: never promise a delivery date you have not looked up.'
RENAMED_TOOL = 'order_status_lookup'

UNREACHABLE_CONFIG: dict[str, object] = {
    'instructions': [{'id': 'toolset:nonexistent', 'instructions': 'this reaches nothing'}],
    'tool_definitions': [{'name': 'not_a_tool', 'description': 'this reaches nothing either'}],
    'settings': {'temperature': 0.1, 'frobnicate': 3},
}
"""One entry per section that can reach nothing here: an unassembled block, an unadvertised tool,
and a settings key the contract has no field for."""

UNREACHABLE_TOKENS = ('toolset:nonexistent', 'not_a_tool', 'frobnicate')


# --- Gating and fixtures ----------------------------------------------------


@pytest.fixture(scope='module')
def platform() -> Platform:
    """The platform this module runs against, or a skip saying what is missing.

    The platform is checked before the provider key, because an unconfigured target is the reason
    this suite skips for everyone who has not deliberately pointed it somewhere.
    """
    target = platform_target()
    require_provider_key('ANTHROPIC_API_KEY')
    return target


@pytest.fixture(scope='module')
def spans(platform: Platform) -> Iterator[SpanCapture]:
    """Configure Logfire against the platform once, and capture what the process exports."""
    capture = SpanCapture()
    configure(platform, capture)
    logfire.instrument_pydantic_ai()
    yield capture
    flush()


@pytest.fixture(autouse=True)
def fresh_process_state(spans: SpanCapture) -> Iterator[None]:
    """Start each test with the once-per-process guards empty and no spans recorded.

    A drop warns once per process and an agent reports itself once per process, both by design, so
    without this a test's outcome would depend on which tests ran before it.
    """
    _forget()
    spans.clear()
    yield
    _forget()


def _forget() -> None:
    """Clear both once-per-process guards: the contract's reporting one and the capability's."""
    reset_warned_messages()
    _agent_control._warned_drops.clear()  # pyright: ignore[reportPrivateUsage]
    _agent_control._reset_config_hint_guard()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture(autouse=True)
def fresh_variable(platform: Platform) -> Iterator[None]:
    """Delete this run's own variable before and after every test.

    Every test publishes the state it is about to check, and none reads state the test before it left
    behind -- so the suite can be run in any order, repeatedly, and after an interrupted run. Deleting
    afterwards as well keeps a failed run from leaving a config on the platform. Nothing here can
    delete a config someone else made: `VARIABLE_NAME` is generated per run.
    """
    _reset_variable(platform)
    yield
    _reset_variable(platform)


def _reset_variable(platform: Platform) -> None:
    platform.delete_variable()
    refresh_variables()


# --- Helpers ----------------------------------------------------------------


def publish(platform: Platform, config: dict[str, object], *, label: str = PRODUCTION_LABEL) -> None:
    """Put one config live at `label`, creating the variable first, and make sure the SDK sees it."""
    if platform.get_variable() is None:
        platform.create_variable()
    platform.publish({label: config})
    refresh_variables()


def baseline_of(spans: SpanCapture) -> AgentConfig:
    """The code baseline the hint span carried, validated against the contract that defines it."""
    return AgentConfig.model_validate_json(spans.hint_baseline_json())


def blocks_of(config: AgentConfig) -> dict[str | None, InstructionBlock]:
    """A baseline's instruction blocks, keyed by the `id` a published config addresses them by."""
    instructions = config.instructions
    assert isinstance(instructions, list), f'a baseline lists its blocks, one per id: {instructions!r}'
    blocks: dict[str | None, InstructionBlock] = {}
    for entry in instructions:
        assert isinstance(entry, InstructionBlock), f'a baseline block carries its id, not bare text: {entry!r}'
        blocks[entry.id] = entry
    return blocks


def unmatched_warnings(messages: list[str]) -> list[str]:
    """The warnings that name one of the entries `UNREACHABLE_CONFIG` cannot apply."""
    return [message for message in messages if any(token in message for token in UNREACHABLE_TOKENS)]


def code_blocks_intact(live: LiveAgent, *, except_for: str | None = None) -> None:
    """Assert every code-defined block is in the prompt with the text the code gives it."""
    for block_id, text in CODE_BLOCKS.items():
        if block_id == except_for:
            continue
        assert live.last.block(block_id) == text, f'{block_id} is not the text the code says'


def hint_span_from_platform(platform: Platform, spans: SpanCapture, *, variable_name: str) -> dict[str, object]:
    """One agent's hint span, read back out of the platform rather than off the local pipeline.

    Polls rather than querying once: ingest is asynchronous, so a query issued the instant after the
    flush can legitimately find nothing yet. `variable_name` belongs to the read-back test that asks,
    so the span this returns cannot be one an earlier test in the run emitted -- which matters,
    because every unconfigured test reports the same baseline under the same reason and matching one
    of those would let a broken exporter pass. The digest is checked too, so what comes back is this
    agent's code rather than some other report filed under the same name.
    """
    flush()
    local = spans.hint_attributes()
    sql = (
        'select span_name, attributes from records '
        f"where span_name = '{HINT_SPAN}' and start_timestamp > now() - interval '1 hour' "
        'order by start_timestamp desc'
    )
    deadline = time.monotonic() + 60
    while True:
        for row in platform.query(sql):
            attributes = row.attributes
            matches = attributes.get('agent_control.variable_name') == variable_name and attributes.get(
                'agent_control.baseline_sha256'
            ) == local.get('agent_control.baseline_sha256')
            if matches:
                return attributes
        if time.monotonic() >= deadline:
            pytest.fail(
                f'no {HINT_SPAN} span for {variable_name} arrived within 60s. The span was emitted -- '
                'that half of this evidence has its own test -- so this is the platform not ingesting '
                'it. See the README on materialized views before believing `show tables`.'
            )
        time.sleep(2)


def assert_hint_attributes_complete(attributes: dict[str, object]) -> None:
    """Assert a hint span carries every attribute the contract promises a consumer."""
    promised = [
        'agent_control.variable_name',
        'agent_control.agent_name',
        'agent_control.framework',
        'agent_control.baseline_source',
        'agent_control.schema_sha256',
        'agent_control.baseline_sha256',
        'agent_control.baseline',
        'agent_control.baseline_reduction',
        'agent_control.baseline_bytes',
        'agent_control.resolution_reason',
        'agent_control.service_name',
        'agent_control.environment',
        'agent_control.service_version',
    ]
    assert [key for key in promised if key not in attributes] == []


# --- 1. An unconfigured agent ------------------------------------------------


def test_an_unconfigured_agent_runs_exactly_as_written(platform: Platform) -> None:
    """Nothing is published, and nothing is different. This is the whole safety story."""
    live = build_agent()

    result = live.run('What is the status of order A-1001?')

    assert result.output
    resolved = live.last.resolved
    assert resolved is not None
    assert (resolved.reason, resolved.label, resolved.version, resolved.sections) == ('code_default', None, None, [])
    assert live.last.block_ids == CODE_BLOCK_ORDER
    assert live.last.tool_names == CODE_TOOL_ORDER
    code_blocks_intact(live)


# --- 2. The hint span --------------------------------------------------------


def test_one_hint_span_per_process_carries_every_promised_attribute(spans: SpanCapture) -> None:
    """The SDK writes no variable: it says what the agent looks like in code, on one span."""
    live = build_agent()
    live.run('Hello.')
    live.run('Hello again.')

    assert len(spans.named(HINT_SPAN)) == 1, 'the hint is emitted once per process per agent'
    attributes = spans.hint_attributes()
    promised = {
        'agent_control.variable_name': VARIABLE_NAME,
        'agent_control.agent_name': AGENT_NAME,
        'agent_control.framework': 'pydantic-ai',
        'agent_control.baseline_source': 'code',
        'agent_control.schema_sha256': SCHEMA_SHA256,
        'agent_control.resolution_reason': 'code_default',
        'agent_control.baseline_reduction': 'none',
        'agent_control.service_name': SERVICE_NAME,
        'agent_control.environment': ENVIRONMENT,
        'agent_control.service_version': SERVICE_VERSION,
    }
    assert {key: attributes.get(key) for key in promised} == promised
    assert attributes.get('agent_control.baseline_bytes') == len(spans.hint_baseline_json().encode())


def test_two_controls_for_one_agent_report_it_once(spans: SpanCapture) -> None:
    """Once per process *per agent*, which is what the docs promise and what a consumer dedupes on.

    Two agents rather than two runs of one, because an agent constructed per request is an ordinary
    shape and each one brings its own `AgentControl`.

    This was an `xfail` and is not any more. The guard was keyed on `variable.logfire_instance`, and
    `Variable.__init__` stores a `with_settings(...)` copy -- a new object per `Variable` -- so a
    second `AgentControl` for one agent missed the guard and reported again, which for an agent built
    per request is one hint span, and one full baseline walk, on every request. It is keyed on the
    Logfire config now, which is the object that really is one per process per project.
    """
    build_agent().run('Hello.')
    build_agent().run('Hello again.')

    assert len(spans.named(HINT_SPAN)) == 1


def test_the_hint_span_arrives_at_the_platform(platform: Platform, spans: SpanCapture) -> None:
    """A span that was emitted is not the same claim as a span that arrived.

    This is the half that needs the platform: a process holding only a span write token can register
    an agent, and the Agent Control page is built from what it sent.
    """
    require_span_read_back(platform)
    # An agent name of its own, so the span queried for below is one only this test emitted. It
    # publishes nothing and no variable is created for it, which is the claim: an agent Logfire holds
    # no config for still reports itself, and needs no write scope to do it.
    agent_name = f'{AGENT_NAME}_reporting'
    live = build_agent(agent_name=agent_name)
    live.run('Hello.')

    attributes = hint_span_from_platform(platform, spans, variable_name=agent_variable_name(agent_name))

    assert_hint_attributes_complete(attributes)
    assert attributes['agent_control.agent_name'] == agent_name
    assert attributes['agent_control.resolution_reason'] == 'code_default'
    assert attributes['agent_control.baseline'] == spans.hint_baseline_json(), (
        'the baseline the platform stored is not the one the process sent'
    )


# --- 3. The baseline ---------------------------------------------------------


def test_the_baseline_describes_every_block_and_the_dynamic_one_without_its_text(spans: SpanCapture) -> None:
    """The baseline is documentation of the code, and a dynamic block contributes only its seam."""
    live = build_agent()
    live.run('Hello.')
    baseline = baseline_of(spans)
    blocks = blocks_of(baseline)

    assert set(blocks) == set(CODE_BLOCK_ORDER)
    for block_id, text in CODE_BLOCKS.items():
        assert blocks[block_id].instructions == text
    dynamic = blocks[DYNAMIC_BLOCK_ID]
    assert dynamic.dynamic is True
    assert dynamic.instructions is None
    # The tenant, the customer id and the date are all in the prompt this run sent. None of them may
    # be in the document the editor shows, or the snapshot would carry one run's data forever. The
    # rendered block is searched for as a whole, so the date looked for is the one the run used.
    serialized = spans.hint_baseline_json()
    leaked = [
        token for token in (*DYNAMIC_DEPS_TOKENS, todays_date(), live.rendered_dynamic[-1]) if token in serialized
    ]
    assert leaked == []


def test_the_baseline_describes_the_model_the_settings_and_every_tool(spans: SpanCapture) -> None:
    """Everything the editor offers to change has to be in the document it shows changes against."""
    live = build_agent()
    live.run('Hello.')
    baseline = baseline_of(spans)

    assert baseline.model == CODE_MODEL
    assert baseline.settings is not None
    assert baseline.settings.model_dump(exclude_none=True) == {'temperature': 0.2, 'max_tokens': 400}
    assert baseline.tool_definitions is not None
    tools = {tool.name: tool for tool in baseline.tool_definitions}
    assert {name: tool.toolset for name, tool in tools.items()} == {
        'lookup_order': 'orders',
        'refund_order': 'orders',
        'search_catalog': 'catalog',
        'lookup_stock': 'catalog',
    }
    # Per-parameter descriptions, because rewriting one is a thing a config can do.
    parameters = tools['search_catalog'].parameters
    assert parameters is not None
    assert parameters['query'].description == "Words from the product name, e.g. 'desk lamp'."


def test_the_baseline_survives_logfire_s_scrubbing(spans: SpanCapture) -> None:
    """A prompt containing `auth` reaches the hint verbatim, and the digest still verifies.

    Logfire's scrubbing is on by default and matches substrings, and the hint's attributes are exempt
    from it (`BaseScrubber.SAFE_KEYS`). Without that exemption `toolset:orders` -- "Order tools are
    authoritative..." -- arrives as `[Scrubbed due to 'auth']`, the editor promotes a redacted prompt
    into the config's `example`, and `baseline_sha256` no longer verifies against the document the
    span carries. This suite runs with scrubbing at its default, so it is the regression test.
    """
    live = build_agent()
    live.run('Hello.')

    blocks = blocks_of(baseline_of(spans))
    assert blocks['toolset:orders'].instructions == CODE_BLOCKS['toolset:orders']
    document = json.loads(spans.hint_baseline_json())
    digest = hashlib.sha256(canonical_json(document)).hexdigest()
    assert spans.hint_attributes().get('agent_control.baseline_sha256') == digest


# --- 4. Creating the config --------------------------------------------------


def test_the_config_can_be_created_from_the_hint_the_way_the_ui_would(platform: Platform, spans: SpanCapture) -> None:
    """Nothing in the SDK creates a config, so this is exactly what the promote flow has to do.

    Three things this pins down, each found the hard way: `kind='agent'` without `display_name` is a
    `400`, the schema stored has to be the contract's own `AGENT_CONFIG_JSON_SCHEMA` rather than one
    derived from the Pydantic model (the UI edits against it and the platform validates later
    versions against it), and the `example` is the hint's baseline byte for byte.
    """
    live = build_agent()
    live.run('Hello.')
    example = spans.hint_baseline_json()

    platform.create_variable(example=example)

    stored = platform.get_variable()
    assert stored is not None
    assert stored.kind == 'agent'
    assert stored.display_name == AGENT_NAME
    assert stored.example == example
    assert stored.schema_document() == AGENT_CONFIG_JSON_SCHEMA


# --- 5. Overriding instruction blocks ----------------------------------------


def test_overriding_the_agent_s_own_block_changes_the_answer_and_nothing_else(platform: Platform) -> None:
    """Not "a managed prompt": this block, by id, and nothing else."""
    publish(platform, {'instructions': [{'id': 'agent', 'instructions': FLAMINGO}]})
    live = build_agent()

    result = live.run('Say hello.')

    assert live.last.block('agent') == FLAMINGO
    assert '<<FLAMINGO>>' in result.output, 'the model did not read the published block'
    code_blocks_intact(live, except_for='agent')
    assert live.last.block_ids == CODE_BLOCK_ORDER, 'an override keeps its block in place'


def test_overriding_one_named_block_leaves_the_agent_s_own_text_alone(platform: Platform) -> None:
    """A named part of a longer prompt is separately addressable, which is the point of naming it."""
    escalation = 'If the customer is angry, apologize once and transfer to a human immediately.'
    publish(platform, {'instructions': [{'id': 'agent:escalation', 'instructions': escalation}]})
    live = build_agent()

    live.run('Say hello.')

    assert live.last.block('agent:escalation') == escalation
    code_blocks_intact(live, except_for='agent:escalation')
    assert live.last.block(DYNAMIC_BLOCK_ID) == live.rendered_dynamic[-1], 'the dynamic block still recomputes'


def test_a_dynamic_block_is_shown_but_never_overridable(platform: Platform) -> None:
    """Pinning it would freeze today's date and silence the tenant the run is serving."""
    frozen = 'Today is 1999-01-01. You are serving the acme storefront for customer cus_1.'
    publish(platform, {'instructions': [{'id': DYNAMIC_BLOCK_ID, 'instructions': frozen}]})

    with pytest.raises(UserError) as error:
        build_agent(on_unmatched='error').run('Say hello.')

    message = str(error.value)
    assert DYNAMIC_BLOCK_ID in message, 'the refusal has to name the block it refused'
    assert 'request' in message.lower() or 'dynamic' in message.lower(), 'and say why'

    live = build_agent(on_unmatched='ignore')
    live.run('Say hello.')
    text = live.last.block(DYNAMIC_BLOCK_ID) or ''
    assert text != frozen
    assert text == live.rendered_dynamic[-1], 'the block is what the code produced, not what was published'
    assert all(token in text for token in DYNAMIC_DEPS_TOKENS), 'and it still reads this run'


# --- 6. Adding a block -------------------------------------------------------


def test_an_added_block_lands_after_the_last_static_block(platform: Platform) -> None:
    """Last in the prompt as written, and still inside the prefix a provider can cache."""
    publish(platform, {'instructions': [ADDED_BLOCK]})
    live = build_agent()

    live.run('Say hello.')

    added = [index for index, (_, _, text) in enumerate(live.last.blocks) if text == ADDED_BLOCK]
    assert len(added) == 1
    index = added[0]
    ids = live.last.block_ids
    assert ids[index] is None, 'nothing addresses an addition, so it carries no id'
    assert index > ids.index('agent'), "an addition is not prepended above the agent's own text"
    last_static = max(position for position, (_, dynamic, _) in enumerate(live.last.blocks) if not dynamic)
    assert index == last_static
    assert index < ids.index(DYNAMIC_BLOCK_ID)


# --- 7. Removing a block, and reverting to code ------------------------------


def test_a_block_can_be_dropped_and_its_text_goes_with_it(platform: Platform) -> None:
    """`instructions: null` removes the block, rather than blanking it."""
    publish(platform, {'instructions': [{'id': 'capability:store_policy', 'instructions': None}]})
    live = build_agent()

    live.run('Say hello.')

    assert 'capability:store_policy' not in live.last.block_ids
    assert CODE_BLOCKS['capability:store_policy'] not in live.last.prompt
    code_blocks_intact(live, except_for='capability:store_policy')


def test_removing_the_section_and_deleting_the_variable_both_revert_to_code(platform: Platform) -> None:
    """The two ways back to the code, and the reason there are two.

    There is no way to remove a published *value* through `/v1`: a `PUT` with an empty `labels`
    object leaves the labels as they were, and there is no per-label delete outside the UI. So going
    back to code is either publishing a config with the section absent, or deleting the variable.
    """
    publish(platform, {'instructions': [{'id': 'agent', 'instructions': FLAMINGO}]})
    live = build_agent()
    live.run('Say hello.')
    assert live.last.block('agent') == FLAMINGO

    publish(platform, {})
    live = build_agent()
    live.run('Say hello.')
    code_blocks_intact(live)
    assert live.last.resolved is not None
    assert live.last.resolved.sections == [], 'an empty config manages nothing'

    _reset_variable(platform)
    live = build_agent()
    live.run('Say hello.')
    code_blocks_intact(live)
    assert live.last.resolved is not None
    assert live.last.resolved.reason == 'code_default'


# --- 8. Renaming a tool ------------------------------------------------------


def test_a_renamed_tool_reaches_the_model_renamed_and_the_code_unchanged(platform: Platform) -> None:
    """The model sees the new name, the code never does, so a name-based policy keeps working."""
    publish(
        platform,
        {'tool_definitions': [{'name': 'lookup_order', 'toolset': 'orders', 'new_name': RENAMED_TOOL}]},
    )
    live = build_agent()

    result = live.run('Where is my order A-1002?')

    first = live.requests[0]
    assert RENAMED_TOOL in first.tool_names
    assert 'lookup_order' not in first.tool_names
    assert 'search_catalog' in first.tool_names, 'the other toolset was left alone'
    called = [
        part.tool_name for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)
    ]
    assert called and set(called) == {RENAMED_TOOL}, 'the model called the managed name, per the history'
    # The other half of the evidence: the code's own function ran, and `ctx.tool_name` inside it is
    # still the code-side name, so a name-based policy in a tool body keeps working through a rename.
    assert [call.function for call in live.tool_calls] == ['lookup_order']
    assert {call.ctx_tool_name for call in live.tool_calls} == {'lookup_order'}
    assert 'transit' in result.output.lower(), 'the answer carries the real tool result'


# --- 9. Descriptions ---------------------------------------------------------


def test_rewritten_descriptions_reach_the_model_and_the_schema_stays_code_owned(platform: Platform) -> None:
    """Logfire edits what the model is told about a tool, never what it does or what it takes."""
    description = 'Search the storefront catalog. Always call this before quoting any price.'
    query_description = 'Product name words only -- never a SKU, never a price.'
    publish(
        platform,
        {
            'tool_definitions': [
                {
                    'name': 'search_catalog',
                    'toolset': 'catalog',
                    'description': description,
                    'parameters': {'query': {'description': query_description}},
                }
            ]
        },
    )
    live = build_agent()

    live.run('How much is the oak desk lamp?')

    _, advertised_description, schema = live.requests[0].tool('search_catalog')
    assert advertised_description == description
    properties = schema['properties']
    assert properties['query']['description'] == query_description
    assert sorted(properties) == ['limit', 'query'], 'the parameters themselves are code-owned'
    assert properties['query']['type'] == 'string'
    assert properties['limit']['default'] == 3, 'including a default the published override never mentions'
    assert live.requests[0].tool('lookup_order')[1] == 'Look up one order belonging to the signed-in customer.'
    assert {call.function for call in live.tool_calls} == {'search_catalog'}, 'the model used what it was told about'


# --- 10. Model override ------------------------------------------------------


def test_a_published_model_overrides_the_code_model(platform: Platform) -> None:
    """A different provider entirely, so the switch is read off the response rather than inferred."""
    require_provider_key('OPENAI_API_KEY')
    publish(platform, {'model': MANAGED_MODEL})
    live = build_agent()

    result = live.run('Say hello in five words.')

    managed_name = MANAGED_MODEL.split(':', 1)[1]
    assert live.last.model_name == managed_name
    assert result.response.model_name is not None
    assert managed_name in result.response.model_name, 'the provider stamped the response with it'
    assert result.output


# --- 11. Settings ------------------------------------------------------------


def test_published_settings_patch_over_the_code_settings(platform: Platform) -> None:
    """A published section is a patch: a key it does not mention keeps the value the code set."""
    publish(
        platform,
        {'settings': {'temperature': 0.0, 'stop_sequences': ['<<STOP>>'], 'parallel_tool_calls': False}},
    )
    live = build_agent()

    result = live.run('Say hello in five words.')

    assert live.last.settings == {
        'temperature': 0.0,  # published over the code's 0.2
        'max_tokens': 400,  # the code's, untouched: the section never mentioned it
        'stop_sequences': ['<<STOP>>'],
        'parallel_tool_calls': False,
    }
    assert result.output, 'and the provider accepted them'


def test_turning_thinking_on_from_logfire_means_publishing_the_temperature_with_it(platform: Platform) -> None:
    """The trap that follows from patch semantics, against the real provider.

    `thinking` published on its own leaves the agent's code-side `temperature: 0.2` in force, and
    Anthropic refuses that pair with a `400`. The patch is this suite's claim and the refusal is the
    provider's, so read a failure accordingly: the settings assertion below going red is a change in
    how a section is applied, and only the `raises` going red is the provider changing its mind.
    """
    publish(platform, {'settings': {'thinking': 'low', 'max_tokens': 2048}})
    live = build_agent()

    with warnings.catch_warnings():
        # A model that drops an unsupported sampling setting warns, and `filterwarnings = error` in
        # the root config would make that the failure rather than the provider's answer.
        warnings.simplefilter('always')
        with pytest.raises(ModelHTTPError) as error:
            live.run('Say hello in five words.')

    assert live.last.settings == {'thinking': 'low', 'max_tokens': 2048, 'temperature': 0.2}
    assert error.value.status_code == 400

    publish(platform, {'settings': {'thinking': 'low', 'temperature': 1.0, 'max_tokens': 8192}})
    live = build_agent()

    assert live.run('Say hello in five words.').output, 'published with a temperature, it works'
    assert live.last.settings.get('temperature') == 1.0


# --- 12. Precedence ----------------------------------------------------------


def test_a_call_site_model_beats_the_published_one_and_the_rest_still_applies(platform: Platform) -> None:
    """Your call site wins for that one run, and nothing else about the config changes."""
    require_provider_key('OPENAI_API_KEY')
    publish(platform, {'model': MANAGED_MODEL, 'instructions': [{'id': 'agent', 'instructions': FLAMINGO}]})

    live = build_agent()
    live.run('Say hello in five words.')
    assert live.last.model_name == MANAGED_MODEL.split(':', 1)[1], 'published beats Agent(model=...)'

    live = build_agent()
    live.run('Say hello in five words.', model=CODE_MODEL)
    assert live.last.model_name == CODE_MODEL.split(':', 1)[1], 'run(model=...) beats the published model'
    assert live.last.block('agent') == FLAMINGO, 'and the rest of the config still applied'


# --- 13. on_unmatched -------------------------------------------------------


def test_entries_that_reach_nothing_warn_once_per_process(platform: Platform) -> None:
    """Visible without stopping anything, and not once per run: the signal survives repetition."""
    publish(platform, UNREACHABLE_CONFIG)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build_agent(on_unmatched='warn').run('Say hello.')
    first = unmatched_warnings([str(warning.message) for warning in caught])

    assert first, 'nothing was reported'
    for token in UNREACHABLE_TOKENS:
        assert any(token in message for message in first), f'{token} was not named'

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        build_agent(on_unmatched='warn').run('Say hello again.')

    assert unmatched_warnings([str(warning.message) for warning in caught]) == []


def test_entries_that_reach_nothing_can_fail_the_run_naming_all_of_them(platform: Platform) -> None:
    """Every section is judged before any of it is reported, so `'error'` names all three at once."""
    publish(platform, UNREACHABLE_CONFIG)

    with pytest.raises(UserError) as error:
        build_agent(on_unmatched='error').run('Say hello.')

    message = str(error.value)
    assert [token for token in UNREACHABLE_TOKENS if token not in message] == []


def test_entries_that_reach_nothing_can_be_ignored(platform: Platform) -> None:
    """Silent, and the run still works."""
    publish(platform, UNREACHABLE_CONFIG)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = build_agent(on_unmatched='ignore').run('Say hello.')

    assert unmatched_warnings([str(warning.message) for warning in caught]) == []
    assert result.output


# --- 14. A newer UI ---------------------------------------------------------


def test_a_section_this_release_never_heard_of_is_reported_by_name(platform: Platform) -> None:
    """A config saved by a newer Logfire must not fail quietly, and its siblings still apply."""
    config: dict[str, object] = {
        'instructions': [{'id': 'agent', 'instructions': FLAMINGO}],
        'orchestration': {'handoffs': [{'to': 'billing_agent'}]},
        'guardrails': ['pii'],
    }
    publish(platform, config)

    with pytest.raises(UserError) as error:
        build_agent(on_unmatched='error').run('Say hello.')

    message = str(error.value)
    assert 'orchestration' in message
    assert 'guardrails' in message

    live = build_agent(on_unmatched='ignore')
    live.run('Say hello.')
    assert live.last.block('agent') == FLAMINGO, 'the sections this release does understand applied'


# --- 15. Resilience ---------------------------------------------------------


def test_a_value_that_is_not_json_leaves_the_agent_on_its_code(platform: Platform) -> None:
    """Nothing about reading a config can take a run down."""
    platform.create_variable()
    platform.publish({PRODUCTION_LABEL: '{"instructions": [[[ this is not json'}, serialized=True)
    refresh_variables()
    live = build_agent()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = live.run('What is the status of order A-1001?')

    assert result.output
    code_blocks_intact(live)
    assert live.last.model_name == CODE_MODEL.split(':', 1)[1]
    assert live.last.resolved is not None
    reported = live.last.resolved.reason in ('validation_error', 'other_error') or any(
        VARIABLE_NAME in str(warning.message) for warning in caught
    )
    assert reported, f'it was silent about it: reason={live.last.resolved.reason!r}'


def test_an_unreachable_platform_leaves_the_agent_on_its_code(platform: Platform, spans: SpanCapture) -> None:
    """A managed config that can take an agent down when the network wobbles is worse than none.

    Pointed at a closed port, so every variables call fails for real rather than being stubbed out.
    """
    with configured_against(platform, spans, base_url='http://127.0.0.1:9'):
        live = build_agent()
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            result = live.run('What is the status of order A-1001?')

    assert result.output
    code_blocks_intact(live)
    assert live.last.model_name == CODE_MODEL.split(':', 1)[1]
    assert live.last.resolved is not None
    assert live.last.resolved.sections == []
    # `reason='code_default'` here is the same reason an agent with nothing published reports, so an
    # operator cannot tell "nobody has configured me" from "I could not reach Logfire" off the
    # resolution alone. The contract names `no_provider` and `other_error` for the distinction.
    assert live.last.resolved.reason == 'code_default'


# --- 16. Labels -------------------------------------------------------------


def test_two_labels_on_one_variable_resolve_separately(platform: Platform) -> None:
    """One variable, two labels: pinning one is what keeps a deployment stable while another moves."""
    platform.create_variable()
    platform.publish(
        {
            PRODUCTION_LABEL: {'instructions': [{'id': 'agent', 'instructions': FLAMINGO}]},
            CANARY_LABEL: {'instructions': [{'id': 'agent', 'instructions': CANARY}]},
        },
        rollout={PRODUCTION_LABEL: 1.0},
    )
    refresh_variables()

    production = build_agent(label=PRODUCTION_LABEL)
    production.run('Say hello.')
    canary = build_agent(label=CANARY_LABEL)
    canary.run('Say hello.')
    unpinned = build_agent(label=None)
    unpinned.run('Say hello.')

    assert production.last.block('agent') == FLAMINGO
    assert canary.last.block('agent') == CANARY
    assert unpinned.last.block('agent') == FLAMINGO, "an unpinned agent follows the variable's rollout"
    for live, label in ((production, PRODUCTION_LABEL), (canary, CANARY_LABEL)):
        assert live.last.resolved is not None
        assert live.last.resolved.label == label
        assert live.last.resolved.version is not None, 'each run reports the version it resolved'


# --- 17. Once per run -------------------------------------------------------


def test_publishing_mid_run_takes_effect_on_the_next_run(platform: Platform) -> None:
    """Every request of a run agrees on the version that produced it, so a trace can say which.

    The publish is fired from inside the first tool call: the model has made one request and a second
    is about to be assembled from the tool's result, which is the moment a per-request resolution
    would change under the run.
    """
    before = 'You are the BEFORE build. Begin every reply with <<BEFORE>>.'
    after = 'You are the AFTER build. Begin every reply with <<AFTER>>.'
    publish(platform, {'instructions': [{'id': 'agent', 'instructions': before}]})
    live = build_agent()
    published: list[str] = []

    def publish_once(_call: ToolCall) -> None:
        if not published:
            published.append(after)
            publish(platform, {'instructions': [{'id': 'agent', 'instructions': after}]})

    live.tool_hooks.append(publish_once)
    live.run('Where is my order A-1002? Then tell me its total.')

    assert published == [after], 'the write has to land mid-run for this to prove anything'
    assert len(live.requests) > 1, 'and the run needs a second request that could have changed'
    assert [request.block('agent') for request in live.requests] == [before] * len(live.requests)
    versions = {request.resolved.version for request in live.requests if request.resolved is not None}
    assert len(versions) == 1

    refresh_variables()
    next_run = build_agent()
    next_run.run('Say hello.')
    assert next_run.last.block('agent') == after


# --- 18. The hint for a configured agent ------------------------------------


def test_a_configured_agent_still_reports_its_code_baseline(platform: Platform, spans: SpanCapture) -> None:
    """What lets the editor tell a stored baseline that matches the code from one the code moved past.

    An agent that reported only while unconfigured would go quiet the moment someone configured it.
    """
    publish(platform, {'instructions': [{'id': 'agent', 'instructions': FLAMINGO}], 'model': CODE_MODEL})
    live = build_agent()

    live.run('Say hello.')

    assert len(spans.named(HINT_SPAN)) == 1
    assert spans.hint_attributes().get('agent_control.resolution_reason') == 'resolved'
    blocks = blocks_of(baseline_of(spans))
    assert blocks['agent'].instructions == CODE_BLOCKS['agent'], 'the baseline describes the code'
    assert live.last.block('agent') == FLAMINGO, 'and the run itself used the published value'


def test_the_configured_hint_span_arrives_at_the_platform(platform: Platform, spans: SpanCapture) -> None:
    """The read-back for a configured agent, so `resolution_reason` reaches Logfire as `'resolved'`.

    Its own test rather than the tail of the one above: gating a skip on the read tokens has to
    happen before anything publishes or makes a model request, and the local half of this claim must
    report green on a platform with no query credentials rather than reporting skipped.
    """
    require_span_read_back(platform)
    # Its own agent name for the same reason as the unconfigured read-back, which means its own
    # variable: created and deleted here, since `fresh_variable` only owns the suite's own.
    agent_name = f'{AGENT_NAME}_reporting_configured'
    variable_name = agent_variable_name(agent_name)
    try:
        platform.create_variable(name=variable_name, display_name=agent_name)
        platform.publish(
            {PRODUCTION_LABEL: {'instructions': [{'id': 'agent', 'instructions': FLAMINGO}]}},
            name=variable_name,
        )
        refresh_variables()
        live = build_agent(agent_name=agent_name)
        live.run('Say hello.')
        assert live.last.block('agent') == FLAMINGO, 'the published value has to have applied'

        attributes = hint_span_from_platform(platform, spans, variable_name=variable_name)
    finally:
        platform.delete_variable(variable_name)
        refresh_variables()

    assert_hint_attributes_complete(attributes)
    assert attributes['agent_control.resolution_reason'] == 'resolved'
