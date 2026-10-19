"""What a published value does to a request, which is the whole of what an adapter delegates."""

from __future__ import annotations

from typing import Any

from pydantic_ai_harness.logfire._contract import (
    MAX_MODEL_FACING_TEXT_LENGTH,
    AgentConfig,
    AgentSupport,
    AppliedInstructions,
    AppliedSettings,
    AppliedTools,
    ApplyIssue,
    Block,
    Destination,
    InstructionBlock,
    ToolDef,
    apply_instructions,
    apply_settings,
    apply_tool_definitions,
)

SETTINGS_ONLY = AgentSupport(sections=frozenset({'settings'}), settings=frozenset({'temperature', 'max_tokens'}))
"""An adapter that can lower two settings and apply no other section."""

AGENT = Block('You are a checkout assistant.', id='agent')
REFUNDS = Block('Always confirm the order total.', id='agent:refunds')
TODAY = Block('Today is 2026-09-09.', id='agent:today', dynamic=True)

WEATHER = ToolDef(
    name='get_weather',
    description='Get the current weather for a city.',
    parameters_json_schema={
        'type': 'object',
        'properties': {'city': {'type': 'string', 'description': 'City to look up.'}},
        'required': ['city'],
    },
    toolset='<agent>',
)
SEARCH = ToolDef(name='search', description='Search the CRM.', toolset='crm')


def config(**sections: Any) -> AgentConfig:
    return AgentConfig.model_validate(sections)


def test_nothing_published_leaves_every_block_alone() -> None:
    blocks = [AGENT, TODAY]
    assert apply_instructions(blocks, AgentConfig()) == AppliedInstructions(blocks=blocks, issues=[])


def test_an_id_replaces_that_blocks_text_and_leaves_its_position_and_flag() -> None:
    applied = apply_instructions(
        [AGENT, REFUNDS, TODAY], config(instructions=[{'id': 'agent:refunds', 'instructions': 'Confirm the total.'}])
    )
    assert applied.blocks == [AGENT, Block('Confirm the total.', id='agent:refunds'), TODAY]


def test_an_id_with_no_text_drops_the_block() -> None:
    assert apply_instructions([AGENT, REFUNDS], config(instructions=[{'id': 'agent:refunds'}])).blocks == [AGENT]


def test_an_added_block_lands_before_the_first_dynamic_one() -> None:
    # The added text has to sit inside the prefix a provider can cache, which is everything before
    # the first block the agent recomputes per request.
    applied = apply_instructions([AGENT, TODAY], config(instructions=['Escalate anything over $500.']))
    assert applied.blocks == [AGENT, Block('Escalate anything over $500.'), TODAY]


def test_the_short_form_is_one_added_block() -> None:
    applied = apply_instructions([AGENT, TODAY], config(instructions='Escalate anything over $500.'))
    assert applied.blocks == [AGENT, Block('Escalate anything over $500.'), TODAY]


def test_added_blocks_are_appended_in_order_when_nothing_is_dynamic() -> None:
    applied = apply_instructions([AGENT], config(instructions=['First.', {'instructions': 'Second.'}]))
    assert applied.blocks == [AGENT, Block('First.'), Block('Second.')]


def test_a_block_with_no_id_cannot_be_addressed_and_passes_through() -> None:
    anonymous = Block('Assembled by a callable nobody named.')
    applied = apply_instructions([anonymous], config(instructions=['Added.']))
    assert applied.blocks == [anonymous, Block('Added.')]


def test_a_dynamic_block_is_not_addressable() -> None:
    applied = apply_instructions([AGENT, TODAY], config(instructions=[{'id': 'agent:today', 'instructions': 'x'}]))
    assert applied.blocks == [AGENT, TODAY]
    assert [(i.section, i.reason, i.instruction_id) for i in applied.issues] == [
        ('instructions', 'dynamic-id', 'agent:today')
    ]
    assert 'which the agent recomputes per request' in applied.issues[0].message


def test_an_id_this_request_does_not_assemble_is_returned_as_an_issue() -> None:
    # And is reported by nothing here: the section plans, and the adapter reports every section at
    # once, so `'error'` fails on the whole request rather than on whichever section came first.
    applied = apply_instructions([AGENT], config(instructions=[{'id': 'toolset:crm', 'instructions': 'x'}]))
    assert applied.blocks == [AGENT]
    assert [(i.section, i.reason, i.instruction_id) for i in applied.issues] == [
        ('instructions', 'unknown-id', 'toolset:crm')
    ]
    assert 'which this request does not assemble' in applied.issues[0].message


def test_two_entries_naming_one_id_keep_the_first_and_report_the_rest() -> None:
    # It used to warn straight from indexing, which made `'ignore'` warn anyway and `'error'` not
    # raise at all -- the one decision the policy never governed.
    applied = apply_instructions(
        [AGENT],
        config(instructions=[{'id': 'agent', 'instructions': 'First.'}, {'id': 'agent', 'instructions': 'Last.'}]),
    )
    assert applied.blocks == [Block('First.', id='agent')]
    assert [(i.section, i.reason, i.instruction_id) for i in applied.issues] == [
        ('instructions', 'duplicate-entry', 'agent')
    ]
    assert "names instruction id 'agent' more than once" in applied.issues[0].message


def test_a_description_override_leaves_the_rest_of_the_schema_alone() -> None:
    patched = ToolDef(
        name='get_weather',
        description='Look up the weather.',
        parameters_json_schema={
            'type': 'object',
            'properties': {'city': {'type': 'string', 'description': "City name, e.g. 'London'"}},
            'required': ['city'],
        },
        toolset='<agent>',
    )
    applied = apply_tool_definitions(
        [WEATHER],
        config(
            tool_definitions=[
                {
                    'name': 'get_weather',
                    'description': 'Look up the weather.',
                    'parameters': {'city': {'description': "City name, e.g. 'London'"}},
                }
            ]
        ),
    )
    assert applied == AppliedTools(
        tools=[patched],
        routes={'get_weather': 'get_weather'},
        forward={('<agent>', 'get_weather'): 'get_weather'},
        reverse={('<agent>', 'get_weather'): 'get_weather'},
        issues=[],
    )


def test_a_patch_on_a_parameter_the_tool_does_not_have_is_reported() -> None:
    # It used to be silent, which from the Logfire UI looked exactly like a patch that applied.
    published = config(tool_definitions=[{'name': 'get_weather', 'parameters': {'unknown': {'description': 'x'}}}])
    applied = apply_tool_definitions([WEATHER], published)
    assert applied.tools == [WEATHER]
    assert [(i.section, i.reason, i.tool, i.parameter, i.toolset) for i in applied.issues] == [
        ('tool_definitions', 'unknown-parameter', 'get_weather', 'unknown', '<agent>')
    ]
    assert "patches parameter 'unknown' of tool 'get_weather' from toolset '<agent>'" in applied.issues[0].message


def test_a_parameter_whose_schema_is_not_an_object_cannot_be_patched() -> None:
    tool = ToolDef(name='odd', parameters_json_schema={'type': 'object', 'properties': {'flag': True}})
    published = config(tool_definitions=[{'name': 'odd', 'parameters': {'flag': {'description': 'x'}}}])
    applied = apply_tool_definitions([tool], published)
    assert applied.tools == [tool]
    assert [i.reason for i in applied.issues] == ['no-patchable-schema']
    assert 'describes that parameter with no schema object' in applied.issues[0].message


def test_an_override_that_changes_nothing_returns_the_definition_it_was_given() -> None:
    applied = apply_tool_definitions(
        [WEATHER],
        config(
            tool_definitions=[
                {
                    'name': 'get_weather',
                    'new_name': 'get_weather',
                    'description': 'Get the current weather for a city.',
                    'parameters': {'city': {'description': None}},
                }
            ]
        ),
    )
    assert applied.tools[0] is WEATHER


def test_a_tool_with_no_patchable_schema_reports_every_parameter_patch() -> None:
    tool = ToolDef(name='ping', parameters_json_schema={'type': 'object'})
    applied = apply_tool_definitions(
        [tool], config(tool_definitions=[{'name': 'ping', 'parameters': {'city': {'description': 'x'}}}])
    )
    assert applied.tools == [tool]
    assert [(i.reason, i.parameter) for i in applied.issues] == [('no-patchable-schema', 'city')]
    assert "patches parameter 'city' of tool 'ping', which has no top-level parameters" in applied.issues[0].message


def test_a_rename_is_advertised_and_routed_back_to_the_code_name() -> None:
    applied = apply_tool_definitions(
        [WEATHER, SEARCH], config(tool_definitions=[{'name': 'get_weather', 'new_name': 'lookup_weather'}])
    )
    assert [tool.name for tool in applied.tools] == ['lookup_weather', 'search']
    assert applied.routes == {'lookup_weather': 'get_weather', 'search': 'search'}


def test_a_rename_onto_another_advertised_name_is_dropped_and_the_rest_of_the_patch_applies() -> None:
    published = config(tool_definitions=[{'name': 'get_weather', 'new_name': 'search', 'description': 'Weather.'}])
    applied = apply_tool_definitions([WEATHER, SEARCH], published)
    assert applied.tools[0] == ToolDef(
        name='get_weather',
        description='Weather.',
        parameters_json_schema=WEATHER.parameters_json_schema,
        toolset='<agent>',
    )
    assert applied.routes == {'get_weather': 'get_weather', 'search': 'search'}
    assert [(i.reason, i.tool) for i in applied.issues] == [('rename-collision', 'get_weather')]
    assert "renames 'get_weather' to 'search', which is already advertised" in applied.issues[0].message


def test_a_rename_onto_a_name_the_adapter_reserved_is_refused() -> None:
    # An OpenAI Agents handoff, or a provider tool the framework adds after this call: outside the
    # editable list, and the core cannot see it unless the adapter says so.
    published = config(tool_definitions=[{'name': 'get_weather', 'new_name': 'transfer_to_billing'}])
    applied = apply_tool_definitions([WEATHER], published, reserved={'transfer_to_billing'})
    assert [tool.name for tool in applied.tools] == ['get_weather']
    assert [i.reason for i in applied.issues] == ['rename-collision']
    assert "renames 'get_weather' to 'transfer_to_billing'" in applied.issues[0].message


def test_names_collide_per_toolset_when_the_runtime_name_carries_the_toolset() -> None:
    # Claude advertises `mcp__<server>__<tool>`, so two servers may each answer to `lookup`.
    crm = ToolDef(name='search', toolset='crm')
    docs = ToolDef(name='lookup', toolset='docs')
    published = config(tool_definitions=[{'name': 'search', 'toolset': 'crm', 'new_name': 'lookup'}])
    applied = apply_tool_definitions([crm, docs], published, collision_scope='toolset')
    assert [tool.name for tool in applied.tools] == ['lookup', 'lookup']
    assert applied.issues == []
    # The flat map cannot tell the two apart, which is exactly why the identity maps exist.
    assert applied.reverse == {('crm', 'lookup'): 'search', ('docs', 'lookup'): 'lookup'}
    assert applied.forward == {('crm', 'search'): 'lookup', ('docs', 'lookup'): 'lookup'}
    assert applied.routes == {'lookup': 'search'}
    # The same rename against one flat namespace is a collision.
    globally = apply_tool_definitions([crm, docs], published)
    assert [tool.name for tool in globally.tools] == ['search', 'lookup']
    assert [i.reason for i in globally.issues] == ['rename-collision']


def test_a_rename_onto_a_name_an_earlier_rename_took_is_dropped_too() -> None:
    applied = apply_tool_definitions(
        [WEATHER, SEARCH],
        config(
            tool_definitions=[
                {'name': 'get_weather', 'new_name': 'lookup'},
                {'name': 'search', 'new_name': 'lookup'},
            ]
        ),
    )
    assert applied.routes == {'lookup': 'get_weather', 'search': 'search'}
    assert [(i.reason, i.tool) for i in applied.issues] == [('rename-collision', 'search')]
    assert "renames 'search' to 'lookup', which is already advertised" in applied.issues[0].message


def test_a_toolset_qualified_override_beats_one_that_only_names_the_tool() -> None:
    crm_search = ToolDef(name='search', description='Search the CRM.', toolset='crm')
    docs_search = ToolDef(name='search', description='Search the docs.', toolset='docs')
    applied = apply_tool_definitions(
        [crm_search, docs_search],
        config(
            tool_definitions=[
                {'name': 'search', 'description': 'Search something.'},
                {'name': 'search', 'toolset': 'crm', 'description': 'Search CRM records.'},
            ]
        ),
    )
    # The unqualified entry is outranked where the narrowed one applies, not unmatched: it reached
    # the docs tool, so nothing is reported.
    assert [tool.description for tool in applied.tools] == ['Search CRM records.', 'Search something.']


def test_an_override_that_matches_no_tool_is_reported() -> None:
    published = config(tool_definitions=[{'name': 'search', 'toolset': 'crm'}])
    applied = apply_tool_definitions([WEATHER], published)
    assert applied.tools == [WEATHER]
    assert [(i.section, i.reason, i.toolset, i.tool) for i in applied.issues] == [
        ('tool_definitions', 'unknown-tool', 'crm', 'search')
    ]
    assert "patches tool 'search' from toolset 'crm', which no toolset advertises" in applied.issues[0].message


def test_two_overrides_naming_one_tool_keep_the_first_and_report_the_rest() -> None:
    applied = apply_tool_definitions(
        [WEATHER],
        config(
            tool_definitions=[
                {'name': 'get_weather', 'description': 'First.'},
                {'name': 'get_weather', 'description': 'Last.'},
            ]
        ),
    )
    assert applied.tools[0].description == 'First.'
    assert [(i.section, i.reason, i.toolset, i.tool) for i in applied.issues] == [
        ('tool_definitions', 'duplicate-entry', None, 'get_weather')
    ]
    assert "names tool 'get_weather' more than once" in applied.issues[0].message


def test_what_an_adapter_declares_it_can_do_and_what_it_leaves_to_the_defaults() -> None:
    # The declaration is adapter capability and nothing else: there is deliberately no way to say
    # anything here about what a given model will accept.
    support = AgentSupport(
        sections=frozenset({'instructions', 'settings'}),
        destinations=(Destination(id='prompt', default=True), Destination(id='preset', accepts_additions=False)),
        settings=frozenset({'temperature'}),
    )
    assert support.precedence == 'exact'
    assert support.resolution_unit == 'run'
    # A destination accepts additions unless the adapter says otherwise, and is the default only if
    # it says so; `AgentSupport` declares no destinations at all for a framework with one prompt.
    assert support.destinations[0] == Destination(id='prompt', default=True, accepts_additions=True)
    assert support.destinations[1] == Destination(id='preset', default=False, accepts_additions=False)
    assert AgentSupport(sections=frozenset()).destinations == ()
    assert AgentSupport(sections=frozenset()).settings == frozenset()


def test_no_settings_published_is_an_empty_patch() -> None:
    assert apply_settings(AgentConfig()) == AppliedSettings(settings={}, issues=[])


def test_only_the_keys_that_were_set_are_in_the_patch() -> None:
    applied = apply_settings(config(settings={'temperature': 0.4, 'stop_sequences': ['STOP']}))
    assert applied == AppliedSettings(settings={'temperature': 0.4, 'stop_sequences': ['STOP']}, issues=[])


def test_a_key_this_contract_has_no_field_for_is_reported() -> None:
    applied = apply_settings(config(settings={'temperature': 0.4, 'service_tier': 'flex'}))
    assert applied.settings == {'temperature': 0.4}
    assert [(i.section, i.reason, i.setting) for i in applied.issues] == [
        ('settings', 'unknown-setting', 'service_tier')
    ]
    assert "sets 'service_tier', which this version of the Agent Control contract" in applied.issues[0].message


def test_a_key_the_adapter_cannot_lower_is_reported_and_left_out() -> None:
    published = config(settings={'temperature': 0.4, 'top_k': 20})
    applied = apply_settings(published, support=SETTINGS_ONLY)
    assert applied.settings == {'temperature': 0.4}
    assert [(i.section, i.reason, i.setting) for i in applied.issues] == [('settings', 'unsupported-setting', 'top_k')]
    assert "sets 'top_k', which this agent framework has no equivalent for" in applied.issues[0].message
    # No support declared is the adapter saying it can lower all of them, which is what every
    # adapter that has not been taught to declare yet is doing.
    assert apply_settings(published).settings == {'temperature': 0.4, 'top_k': 20}


def test_a_section_the_adapter_cannot_apply_is_reported_once() -> None:
    # The editor can grey out a section from this; nothing else in the contract can tell it that a
    # `model` an adapter cannot switch is not worth publishing.
    published = config(model='openai:gpt-5.6-sol', instructions=['Be brief.'], settings={'temperature': 0.4})
    applied = apply_settings(published, support=SETTINGS_ONLY)
    assert applied.settings == {'temperature': 0.4}
    assert [(i.section, i.reason) for i in applied.issues] == [
        ('instructions', 'unsupported-section'),
        ('model', 'unsupported-section'),
    ]
    assert "publishes a 'model' section, which this agent framework has no way to apply" in applied.issues[1].message
    # Declaring the section is what silences it, and it is per section rather than per entry.
    everything = AgentSupport(
        sections=frozenset({'instructions', 'model', 'settings'}), settings=frozenset({'temperature'})
    )
    assert apply_settings(published, support=everything).issues == []


def test_a_settings_section_the_adapter_cannot_apply_yields_no_patch() -> None:
    # Handing back the patch anyway would be this helper contradicting the declaration it was just
    # given, and reporting each key again would bury the one report that matters.
    support = AgentSupport(sections=frozenset({'instructions'}), settings=frozenset({'temperature'}))
    applied = apply_settings(config(settings={'temperature': 0.4}), support=support)
    assert applied.settings == {}
    assert [(i.section, i.reason) for i in applied.issues] == [('settings', 'unsupported-section')]


def test_a_top_level_key_this_release_has_no_section_for_is_reported() -> None:
    # The openness is deliberate -- it is what lets a future section be published against an older
    # SDK -- but a drop nobody hears about is a silently degraded agent.
    published = config(instructions=['Be brief.'], mcp_servers=[{'name': 'crm'}], skills=[])
    assert published.unrecognized == ('mcp_servers', 'skills')
    applied = apply_settings(published)
    assert [(i.section, i.reason) for i in applied.issues] == [
        ('mcp_servers', 'unknown-section'),
        ('skills', 'unknown-section'),
    ]
    assert "publishes a 'mcp_servers' section, which this version of the Agent Control contract" in (
        applied.issues[0].message
    )


def test_a_config_built_in_code_remembers_no_unrecognized_sections() -> None:
    # A baseline is built from keyword arguments rather than from a mapping, and an adapter's own
    # extra key would not be something anyone published.
    assert AgentConfig(model='openai:gpt-5.6-sol').unrecognized == ()


def test_re_validating_a_parsed_config_does_not_forget_what_it_remembered() -> None:
    # An adapter that hands a config back through validation -- a test, a migration script -- gets a
    # model instance rather than a mapping, which has no keys to re-read but has already read them.
    published = config(instructions=['Be brief.'], mcp_servers=[])
    assert AgentConfig.model_validate(published).unrecognized == ('mcp_servers',)


def test_a_timeout_that_is_not_a_request_budget_is_dropped_rather_than_clamped() -> None:
    applied = apply_settings(config(settings={'timeout': -1, 'temperature': 0.4}))
    assert applied.settings == {'temperature': 0.4}
    assert [(i.section, i.reason, i.setting) for i in applied.issues] == [
        ('settings', 'unrepresentable-timeout', 'timeout')
    ]


def test_a_setting_the_provider_dropped_is_an_issue_the_core_needs_no_sdk_knowledge_for() -> None:
    # Discovered after the request, by an adapter reading whatever its own SDK reports, so the core
    # takes the two things every one of those shapes carries.
    issue = ApplyIssue.dropped_by_provider('top_k', 'unsupported by this model')
    assert (issue.section, issue.reason, issue.setting) == ('settings', 'dropped-by-provider', 'top_k')
    assert issue.message == (
        "Managed agent config sets 'top_k', which the provider did not apply -- unsupported by this "
        'model; that key had no effect on the request.'
    )


def test_a_config_assembled_in_code_may_hold_bare_strings_in_its_list() -> None:
    # Assignment is not re-validated, so an adapter building a config by hand -- a test, a migration
    # script -- can leave the shorthand in place; a bare entry means the same thing either way.
    published = AgentConfig()
    published.instructions = ['Escalate anything over $500.']
    assert apply_instructions([AGENT], published).blocks == [AGENT, Block('Escalate anything over $500.')]


def test_text_past_the_budget_is_refused_rather_than_truncated() -> None:
    # The budget belongs to the contract rather than to one language's model. Parsing a published
    # value enforces it, and `InstructionBlock` enforces it again on anything validated -- but the
    # apply step is the shared algorithm both cores implement, and in TypeScript, where the parsed
    # shape is a plain interface with no runtime bound, this is where an oversized entry is caught.
    # `model_construct` is the only way to get one past both of Python's checks.
    published = AgentConfig()
    published.instructions = [
        InstructionBlock.model_construct(id='agent', instructions='a' * (MAX_MODEL_FACING_TEXT_LENGTH + 1)),
        InstructionBlock.model_construct(instructions='b' * (MAX_MODEL_FACING_TEXT_LENGTH + 1)),
    ]
    applied = apply_instructions([AGENT], published)
    assert applied.blocks == [AGENT]
    assert [(i.reason, i.instruction_id) for i in applied.issues] == [
        ('oversized-text', 'agent'),
        ('oversized-text', None),
    ]
    assert 'past the 65536-character limit' in applied.issues[0].message
