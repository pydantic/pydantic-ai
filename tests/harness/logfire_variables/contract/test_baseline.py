"""The baseline describes the agent as written, which is what the Logfire editor diffs against."""

from __future__ import annotations

import json

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness.logfire._contract import AgentConfig, Block, InstructionBlock, ToolDef, build_baseline


def published(baseline: AgentConfig) -> str:
    """Exactly the bytes `AgentControl.report_baseline` carries on the config-hint span.

    Nothing writes a variable any more, so this is what the hint's `agent_control.baseline` holds and
    what Logfire stores as the code default when someone promotes it -- not what an SDK wrote there.
    """
    return json.dumps(baseline.model_dump(exclude_none=True), indent=2)


def test_the_baseline_an_agent_publishes() -> None:
    baseline = build_baseline(
        instructions=[
            Block('You are a concise checkout assistant.', id='agent'),
            Block('Always confirm the order total.', id='agent:refunds'),
            Block('Today is 2026-09-09.', id='agent:today', dynamic=True),
        ],
        model='anthropic:claude-fable-5-1',
        settings={'temperature': 0.1, 'extra_headers': {'authorization': 'secret'}},
        tools=[
            ToolDef(
                name='get_weather',
                description='Get the current weather for a city.',
                parameters_json_schema={
                    'type': 'object',
                    'properties': {'city': {'type': 'string', 'description': 'City to look up.'}},
                    'required': ['city'],
                },
                toolset='<agent>',
            )
        ],
    )
    assert published(baseline) == snapshot("""\
{
  "instructions": [
    {
      "id": "agent",
      "instructions": "You are a concise checkout assistant.",
      "dynamic": false
    },
    {
      "id": "agent:refunds",
      "instructions": "Always confirm the order total.",
      "dynamic": false
    },
    {
      "id": "agent:today",
      "dynamic": true
    }
  ],
  "model": "anthropic:claude-fable-5-1",
  "settings": {
    "temperature": 0.1
  },
  "tool_definitions": [
    {
      "name": "get_weather",
      "description": "Get the current weather for a city.",
      "parameters": {
        "city": {
          "description": "City to look up."
        }
      },
      "toolset": "<agent>"
    }
  ]
}\
""")


def test_a_dynamic_block_publishes_its_seam_and_never_its_text() -> None:
    baseline = build_baseline(instructions=[Block('Tenant: ACME. User: u_42.', id='agent:tenant', dynamic=True)])
    assert published(baseline) == snapshot("""\
{
  "instructions": [
    {
      "id": "agent:tenant",
      "dynamic": true
    }
  ]
}\
""")


def test_a_dynamic_block_with_nothing_to_key_it_on_is_left_out() -> None:
    # An editor could neither show it nor offer an override for it, and its text is not publishable.
    assert build_baseline(instructions=[Block('Computed, unnamed.', dynamic=True)]) == AgentConfig()


def test_a_blank_static_block_is_left_out() -> None:
    assert build_baseline(instructions=[Block('   ', id='agent:empty')]) == AgentConfig()


def test_a_static_block_with_no_id_is_published_as_unaddressable() -> None:
    baseline = build_baseline(instructions=[Block('Assembled by a callable nobody named.')])
    assert published(baseline) == snapshot("""\
{
  "instructions": [
    {
      "instructions": "Assembled by a callable nobody named.",
      "dynamic": false
    }
  ]
}\
""")


def test_every_top_level_parameter_is_listed_so_an_undocumented_one_can_be_described() -> None:
    # The empty entry is the point: a parameter with no description is exactly the one someone wants
    # to describe from Logfire, and listing only the documented ones would hide it from the editor.
    baseline = build_baseline(
        tools=[
            ToolDef(name='ping', parameters_json_schema={'type': 'object', 'properties': {'host': {'type': 'string'}}}),
            ToolDef(name='pong', parameters_json_schema={'type': 'object'}),
            # A framework that hands over something other than a schema per property: the parameter
            # still exists, and is still the editor's to describe.
            ToolDef(name='pang', parameters_json_schema={'type': 'object', 'properties': {'host': 'a string'}}),
        ]
    )
    assert published(baseline) == snapshot("""\
{
  "tool_definitions": [
    {
      "name": "ping",
      "parameters": {
        "host": {}
      }
    },
    {
      "name": "pong"
    },
    {
      "name": "pang",
      "parameters": {
        "host": {}
      }
    }
  ]
}\
""")


def test_settings_are_kept_to_the_canonical_keys() -> None:
    # A provider-specific setting still applies to the agent; it is just not part of the contract,
    # and the baseline is readable by every member of the Logfire project.
    baseline = build_baseline(settings={'temperature': 0.2, 'openai_reasoning_summary': 'detailed'})
    assert published(baseline) == snapshot("""\
{
  "settings": {
    "temperature": 0.2
  }
}\
""")


def test_a_malformed_code_setting_does_not_stop_the_baseline() -> None:
    with pytest.warns(UserWarning, match="The agent runs with timeout='soon', which the Agent Control contract"):
        baseline = build_baseline(settings={'timeout': 'soon', 'max_tokens': 1024})
    assert published(baseline) == snapshot("""\
{
  "settings": {
    "max_tokens": 1024
  }
}\
""")


def test_an_effort_level_the_contract_has_no_word_for_is_omitted_not_approximated() -> None:
    # `'max'` is not `'xhigh'`, and a baseline that said so would describe the code as doing
    # something it does not do, in the one artifact the editor presents as the truth about the code.
    with pytest.warns(UserWarning, match="The agent runs with thinking='max'"):
        baseline = build_baseline(settings={'thinking': 'max', 'temperature': 0.3})
    assert published(baseline) == snapshot("""\
{
  "settings": {
    "temperature": 0.3
  }
}\
""")


def test_a_timeout_the_contract_cannot_hold_is_omitted_and_reported() -> None:
    with pytest.warns(UserWarning, match='request timeout of -1 seconds, which the Agent Control contract cannot'):
        assert build_baseline(settings={'timeout': -1}) == AgentConfig()
    with pytest.warns(UserWarning, match='request timeout of inf seconds'):
        assert build_baseline(settings={'timeout': float('inf')}) == AgentConfig()


def test_code_side_settings_are_read_leniently_because_they_are_not_json() -> None:
    # An int for a float and a tuple of stop sequences describe the agent perfectly well; the same
    # shapes arriving over the wire would be a UI writing what the stored schema says it may not.
    baseline = build_baseline(settings={'temperature': 1, 'stop_sequences': ('STOP', 'HALT')})
    assert baseline.settings is not None
    assert (baseline.settings.temperature, baseline.settings.stop_sequences) == (1.0, ['STOP', 'HALT'])


def test_an_agent_with_nothing_to_describe_publishes_an_empty_config() -> None:
    assert published(build_baseline()) == snapshot('{}')


def test_a_code_side_timeout_is_judged_after_coercion_not_before() -> None:
    # A framework holding its timeout as a string validates to a number, and it is that number the
    # baseline would publish, so it is that number the contract has to judge.
    with pytest.warns(UserWarning, match="request timeout of '-1' seconds"):
        assert build_baseline(settings={'timeout': '-1'}) == AgentConfig()
    baseline = build_baseline(settings={'timeout': '5'})
    assert baseline.settings is not None and baseline.settings.timeout == 5.0


def test_an_oversized_code_side_block_is_left_out_rather_than_raised() -> None:
    # `build_baseline` runs on the request path in an observing adapter, and every other code-side
    # value the contract cannot hold is reported and skipped; this one must not be the exception.
    with pytest.warns(UserWarning, match='instruction block of 70000 characters'):
        baseline = build_baseline(instructions=[Block('a' * 70000, id='agent'), Block('Be concise.', id='agent:style')])
    assert baseline.instructions is not None
    assert [entry.id for entry in baseline.instructions if isinstance(entry, InstructionBlock)] == ['agent:style']


def test_an_adapter_block_with_an_empty_id_is_reported_rather_than_raised() -> None:
    # `Block` is the adapter's own dataclass and constrains nothing, while the contract's `id` is a
    # non-empty string. `build_baseline` runs on the request path, so this cannot be the one thing
    # that raises out of it.
    with pytest.warns(UserWarning, match='publishing that block without an id'):
        baseline = build_baseline(instructions=[Block('You are a checkout assistant.', id='')])
    assert baseline.instructions is not None
    [entry] = baseline.instructions
    assert isinstance(entry, InstructionBlock) and entry.id is None


def test_a_dynamic_adapter_block_with_an_empty_id_is_left_out_like_one_with_none() -> None:
    # Nothing to key it on, so the editor could neither show it nor address it -- and the warning
    # says that, rather than the static branch's "publishing that block without an id".
    with pytest.warns(UserWarning, match='leaving it out of the published baseline entirely'):
        assert build_baseline(instructions=[Block('rendered', id='', dynamic=True)]) == AgentConfig()


def test_an_adapter_tool_with_an_empty_name_is_reported_rather_than_raised() -> None:
    # No managed override could address it, and the model could not call it; the request path must
    # not be where that adapter bug surfaces.
    with pytest.warns(UserWarning, match='tool whose name is the empty string'):
        baseline = build_baseline(tools=[ToolDef(''), ToolDef('get_weather', 'Get the weather.')])
    assert baseline.tool_definitions is not None
    assert [override.name for override in baseline.tool_definitions] == ['get_weather']
