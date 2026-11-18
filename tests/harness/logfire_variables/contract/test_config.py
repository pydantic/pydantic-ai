"""A published value degrades one entry at a time, never the whole config."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from pydantic_ai_harness.logfire._contract import (
    AgentConfig,
    AgentConfigSettings,
    InstructionBlock,
    ToolDefinitionOverride,
)
from pydantic_ai_harness.logfire._contract._schema import MAX_MODEL_FACING_TEXT_LENGTH


def test_a_bare_string_is_one_added_block_and_keeps_its_shape() -> None:
    config = AgentConfig.model_validate({'instructions': 'Be brief.'})
    assert config.instructions == 'Be brief.'


def test_every_section_is_optional_and_absent_means_code() -> None:
    config = AgentConfig()
    assert (config.instructions, config.model, config.settings, config.tool_definitions) == (None, None, None, None)


def test_an_invalid_instruction_entry_is_dropped_and_its_siblings_survive() -> None:
    with pytest.warns(UserWarning, match='Managed instruction entry 12 is invalid -- entry=12 .*; ignoring that entry'):
        config = AgentConfig.model_validate({'instructions': ['Keep me.', 12], 'model': 'openai:gpt-5.6-sol'})
    assert config.instructions == [InstructionBlock(instructions='Keep me.')]
    assert config.model == 'openai:gpt-5.6-sol'


def test_an_entry_that_says_neither_what_nor_where_is_dropped() -> None:
    with pytest.warns(UserWarning, match='has neither an `id` to address nor text to add'):
        config = AgentConfig.model_validate({'instructions': [{'dynamic': True}, 'Keep me.']})
    assert config.instructions == [InstructionBlock(instructions='Keep me.')]


def test_an_empty_instruction_section_is_not_a_way_to_blank_the_prompt() -> None:
    with pytest.warns(UserWarning, match="Managed instructions section is invalid -- instructions=''"):
        config = AgentConfig.model_validate({'instructions': '', 'model': 'openai:gpt-5.6-sol'})
    assert config.instructions is None
    assert config.model == 'openai:gpt-5.6-sol'


def test_an_oversized_instruction_section_is_dropped() -> None:
    with pytest.warns(UserWarning, match='exceeding the 65536-character limit'):
        config = AgentConfig.model_validate({'instructions': 'x' * (MAX_MODEL_FACING_TEXT_LENGTH + 1)})
    assert config.instructions is None


def test_the_length_limit_is_the_total_across_entries() -> None:
    # Text that is under the limit alone but does not fit in what the entries before it left.
    entries = ['x' * (MAX_MODEL_FACING_TEXT_LENGTH - 1), 'yy', 'f']
    with pytest.warns(UserWarning, match='does not fit in the 1 remaining of the 65536-character limit'):
        config = AgentConfig.model_validate({'instructions': entries})
    assert config.instructions == [InstructionBlock(instructions=entries[0]), InstructionBlock(instructions='f')]


def test_an_instructions_section_that_is_not_a_string_or_a_list_is_dropped() -> None:
    with pytest.warns(UserWarning, match='Managed instructions section has invalid container'):
        config = AgentConfig.model_validate({'instructions': {'id': 'agent'}, 'model': 'openai:gpt-5.6-sol'})
    assert config.instructions is None
    assert config.model == 'openai:gpt-5.6-sol'


def test_an_invalid_tool_override_is_dropped_and_its_siblings_survive() -> None:
    with pytest.warns(UserWarning, match='Managed tool definition override .* is invalid -- name='):
        config = AgentConfig.model_validate(
            {'tool_definitions': [{'name': ''}, {'name': 'search', 'description': 'Search.'}]}
        )
    assert config.tool_definitions == [ToolDefinitionOverride(name='search', description='Search.')]


def test_a_tool_definitions_section_that_is_not_a_list_is_dropped() -> None:
    with pytest.warns(UserWarning, match='Managed tool definitions section has invalid container'):
        config = AgentConfig.model_validate({'tool_definitions': {'name': 'search'}, 'model': 'openai:gpt-5.6-sol'})
    assert config.tool_definitions is None
    assert config.model == 'openai:gpt-5.6-sol'


def test_a_settings_section_that_is_not_an_object_is_dropped() -> None:
    with pytest.warns(UserWarning, match='Managed settings section has invalid container'):
        config = AgentConfig.model_validate({'settings': 'hot', 'model': 'openai:gpt-5.6-sol'})
    assert config.settings is None
    assert config.model == 'openai:gpt-5.6-sol'


def test_a_settings_section_may_be_given_as_the_model_itself() -> None:
    settings = AgentConfigSettings(temperature=0.4)
    assert AgentConfig(settings=settings).settings is settings


def test_an_unrecognized_settings_key_is_remembered_rather_than_reported_from_validation() -> None:
    config = AgentConfig.model_validate({'settings': {'temperature': 0.4, 'service_tier': 'flex'}})
    assert config.settings is not None
    assert config.settings.temperature == 0.4
    assert config.settings.unrecognized == ('service_tier',)


def test_a_setting_value_a_newer_ui_wrote_costs_only_that_setting() -> None:
    with pytest.warns(UserWarning, match="sets 'thinking' to 'ultra', which this version of the SDK does not"):
        config = AgentConfig.model_validate({'settings': {'thinking': 'ultra', 'temperature': 0.4}})
    assert config.settings == AgentConfigSettings(temperature=0.4)


def test_a_malformed_setting_value_costs_only_that_setting() -> None:
    with pytest.warns(UserWarning, match="setting 'temperature' has invalid value 'hot'"):
        config = AgentConfig.model_validate({'settings': {'temperature': 'hot', 'max_tokens': 100}})
    assert config.settings == AgentConfigSettings(max_tokens=100)


def test_settings_that_are_not_an_object_are_refused_rather_than_half_read() -> None:
    # The per-key leniency is for keys and values inside a settings object. Something that is not an
    # object has no key to keep, so the model refuses it and the enclosing config drops the section.
    with pytest.raises(ValidationError):
        AgentConfigSettings.model_validate(['temperature'])


def test_the_same_warning_is_emitted_once_per_process() -> None:
    value: dict[str, Any] = {'settings': {'temperature': 'hot'}}
    with pytest.warns(UserWarning, match="setting 'temperature' has invalid value 'hot'") as first:
        AgentConfig.model_validate(value)
    assert len(first) == 1
    # A second resolution of the same published value says nothing: the config is read on every
    # request, and a per-request warning would bury the signal under its own repetition.
    AgentConfig.model_validate(value)


def test_an_empty_model_is_refused_because_it_would_take_the_agent_down() -> None:
    # `''` is a model named `''`, which every framework rejects on every request. It costs only the
    # model section, so the instructions someone published alongside it still apply.
    with pytest.warns(UserWarning, match="selects invalid model ''"):
        config = AgentConfig.model_validate({'model': '', 'instructions': 'Be brief.'})
    assert (config.model, config.instructions) == (None, 'Be brief.')


def test_a_model_of_the_wrong_type_costs_only_the_model_section() -> None:
    with pytest.warns(UserWarning, match='selects invalid model 42'):
        config = AgentConfig.model_validate({'model': 42, 'settings': {'temperature': 0.4}})
    assert config.model is None
    assert config.settings is not None and config.settings.temperature == 0.4


def test_the_instruction_budget_is_charged_to_surviving_entries_only() -> None:
    # An entry dropped for being malformed adds nothing to the request, so charging it would let one
    # bad entry shrink the budget for the good ones.
    half = 'a' * 40_000
    with pytest.warns(UserWarning, match='Managed instruction entry'):
        config = AgentConfig.model_validate({'instructions': [{'id': 42, 'instructions': half}, 'b' * 40_000]})
    assert config.instructions == [InstructionBlock(instructions='b' * 40_000)]


def test_a_number_written_as_a_string_is_not_coerced_into_a_setting() -> None:
    # The stored schema says `number`, so `'0.4'` is something a UI may not write; accepting it here
    # would apply a value the TypeScript core drops.
    with pytest.warns(UserWarning, match="setting 'temperature' has invalid value '0.4'"):
        config = AgentConfig.model_validate({'settings': {'temperature': '0.4', 'max_tokens': 2048}})
    assert config.settings is not None
    assert (config.settings.temperature, config.settings.max_tokens) == (None, 2048)


def test_an_integer_is_a_number_because_json_writes_1_for_1_point_0() -> None:
    config = AgentConfig.model_validate({'settings': {'temperature': 1}})
    assert config.settings is not None and config.settings.temperature == 1.0


def test_a_number_is_not_a_boolean() -> None:
    with pytest.warns(UserWarning, match="setting 'parallel_tool_calls' has invalid value 1"):
        config = AgentConfig.model_validate({'settings': {'parallel_tool_calls': 1}})
    assert config.settings is not None and config.settings.parallel_tool_calls is None


def test_an_empty_instruction_id_is_never_a_value() -> None:
    # The stored JSON schema requires a non-empty id, so accepting one here would build baselines the
    # Logfire backend rejects on write, and keep published entries that address nothing.
    with pytest.warns(UserWarning, match="is invalid -- id=''"):
        config = AgentConfig.model_validate(
            {'instructions': [{'id': '', 'instructions': 'x'}, {'id': 'agent', 'instructions': 'y'}]}
        )
    assert [entry.id for entry in (config.instructions or []) if isinstance(entry, InstructionBlock)] == ['agent']
