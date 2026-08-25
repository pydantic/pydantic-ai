from dataclasses import replace
from typing import Literal

import pytest
from pydantic import TypeAdapter

from pydantic_ai.models import ModelRequestParameters, ToolDefinition
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    ImageGenerationTool,
    MCPServerTool,
    MemoryTool,
    WebFetchTool,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai.output import StructuredOutputMode

from .._inline_snapshot import snapshot

ta = TypeAdapter(ModelRequestParameters)


def test_model_request_parameters_are_serializable():
    params = ModelRequestParameters(
        function_tools=[],
        native_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[],
        output_object=None,
    )
    dumped = ta.dump_python(params)
    assert dumped == snapshot(
        {
            'function_tools': [],
            'native_tools': [],
            'tool_visibility': None,
            'revealed_tool_names': set(),
            'deferred_capability_ids': set(),
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [],
            'prompted_output_template': None,
            'allow_text_output': True,
            'allow_image_output': False,
            'instruction_parts': None,
            'thinking': None,
        }
    )
    assert ta.validate_python(dumped) == params

    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='test')],
        native_tools=[
            WebSearchTool(user_location=WebSearchUserLocation(city='New York', country='US')),
            CodeExecutionTool(),
            WebFetchTool(),
            ImageGenerationTool(size='1024x1024'),
            MemoryTool(),
            MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'),
            MCPServerTool(id='github', url='https://api.githubcopilot.com/mcp'),
        ],
        output_mode='text',
        allow_text_output=True,
        output_tools=[ToolDefinition(name='final_result')],
        output_object=None,
    )
    dumped = ta.dump_python(params)
    assert dumped == snapshot(
        {
            'function_tools': [
                {
                    'name': 'test',
                    'parameters_json_schema': {'type': 'object', 'properties': {}},
                    'description': None,
                    'outer_typed_dict_key': None,
                    'strict': None,
                    'sequential': False,
                    'kind': 'function',
                    'metadata': None,
                    'timeout': None,
                    'defer_loading': False,
                    'toolset_id': None,
                    'unless_native': None,
                    'with_native': None,
                    'tool_kind': None,
                    'return_schema': None,
                    'include_return_schema': None,
                    'capability_id': None,
                }
            ],
            'native_tools': [
                {
                    'kind': 'web_search',
                    'optional': False,
                    'search_context_size': 'medium',
                    'user_location': {'city': 'New York', 'country': 'US'},
                    'blocked_domains': None,
                    'allowed_domains': None,
                    'max_uses': None,
                    'external_web_access': None,
                },
                {'kind': 'code_execution', 'optional': False, 'files': None},
                {
                    'kind': 'web_fetch',
                    'optional': False,
                    'max_uses': None,
                    'allowed_domains': None,
                    'blocked_domains': None,
                    'enable_citations': False,
                    'max_content_tokens': None,
                },
                {
                    'kind': 'image_generation',
                    'optional': False,
                    'action': 'auto',
                    'background': 'auto',
                    'input_fidelity': None,
                    'moderation': 'auto',
                    'model': None,
                    'output_compression': None,
                    'output_format': None,
                    'partial_images': 0,
                    'quality': 'auto',
                    'size': '1024x1024',
                    'aspect_ratio': None,
                },
                {'kind': 'memory', 'optional': False},
                {
                    'kind': 'mcp_server',
                    'optional': False,
                    'id': 'deepwiki',
                    'url': 'https://mcp.deepwiki.com/mcp',
                    'authorization_token': None,
                    'description': None,
                    'allowed_tools': None,
                    'headers': None,
                },
                {
                    'kind': 'mcp_server',
                    'optional': False,
                    'id': 'github',
                    'url': 'https://api.githubcopilot.com/mcp',
                    'authorization_token': None,
                    'description': None,
                    'allowed_tools': None,
                    'headers': None,
                },
            ],
            'tool_visibility': None,
            'revealed_tool_names': set(),
            'deferred_capability_ids': set(),
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [
                {
                    'name': 'final_result',
                    'parameters_json_schema': {'type': 'object', 'properties': {}},
                    'description': None,
                    'outer_typed_dict_key': None,
                    'strict': None,
                    'sequential': False,
                    'kind': 'function',
                    'metadata': None,
                    'timeout': None,
                    'defer_loading': False,
                    'toolset_id': None,
                    'unless_native': None,
                    'with_native': None,
                    'tool_kind': None,
                    'return_schema': None,
                    'include_return_schema': None,
                    'capability_id': None,
                }
            ],
            'prompted_output_template': None,
            'allow_text_output': True,
            'allow_image_output': False,
            'instruction_parts': None,
            'thinking': None,
        }
    )
    assert ta.validate_python(dumped) == params


def test_request_visibility_state_survives_serialization_but_stays_out_of_repr():
    """Visibility state has to cross a durable-execution boundary, and stay out of the repr.

    Temporal hands the model activity a `_RequestParams` carrying the whole
    `ModelRequestParameters`, serialized by the pydantic data converter, and the adapters read both
    sets on the far side — Anthropic decides there whether the corpus is capability-only. Excluding
    them from serialization would deliver empty state to every durable run, which nothing that
    stays inside one process would notice. `revealed_tool_names` stays out of the repr via
    `repr=False`; `tool_visibility` relies on the no-defaults repr instead — omitted while `None`
    (authored), visible once resolved — so object snapshots can see resolution state and never
    mismatch invisibly.
    """
    params = ModelRequestParameters(revealed_tool_names={'deferred_tool'})

    round_tripped = ta.validate_python(ta.dump_python(params, mode='json'))

    assert round_tripped.revealed_tool_names == {'deferred_tool'}
    assert repr(params) == snapshot('ModelRequestParameters(function_tools=[], native_tools=[], output_tools=[])')
    assert repr(replace(params, tool_visibility={'t': 'visible'})) == snapshot(
        "ModelRequestParameters(function_tools=[], native_tools=[], tool_visibility={'t': 'visible'}, output_tools=[])"
    )


@pytest.mark.parametrize('visibility', ['visible', 'deferred', 'withheld', 'via_history'])
def test_tool_visibility_round_trip_and_equality(
    visibility: Literal['visible', 'deferred', 'withheld', 'via_history'],
):
    params = ModelRequestParameters(function_tools=[ToolDefinition(name='t')], tool_visibility={'t': visibility})

    dumped = ta.dump_python(params, mode='json')
    round_tripped = ta.validate_python(dumped)
    assert round_tripped.tool_visibility == {'t': visibility}

    del dumped['tool_visibility']
    old_payload = ta.validate_python(dumped)
    assert old_payload.tool_visibility is None

    assert ModelRequestParameters(tool_visibility={'t': visibility}) == ModelRequestParameters(
        tool_visibility={'t': visibility}
    )
    assert ModelRequestParameters(tool_visibility={'t': visibility}) != ModelRequestParameters()


def test_visibility_of_unresolved_parameters_uses_authored_deferral() -> None:
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='hidden', defer_loading=True), ToolDefinition(name='visible')]
    )
    assert params.visibility_of('hidden') == 'withheld'
    assert params.visibility_of('visible') == 'visible'
    assert params.visibility_of('unknown') == 'visible'


@pytest.mark.parametrize(
    'output_mode, expected_allow_text',
    [
        ('tool', False),
        ('native', True),
        ('prompted', True),
    ],
)
def test_with_default_output_mode(output_mode: StructuredOutputMode, expected_allow_text: bool):
    params = ModelRequestParameters(output_mode='auto', allow_text_output=True)
    resolved = params.with_default_output_mode(output_mode)
    assert resolved.output_mode == output_mode
    assert resolved.allow_text_output == expected_allow_text


def test_with_default_output_mode_noop_when_not_auto():
    params = ModelRequestParameters(output_mode='tool', allow_text_output=False)
    resolved = params.with_default_output_mode('native')
    assert resolved is params


def test_with_default_output_mode_overrides_allow_text():
    params = ModelRequestParameters(output_mode='auto', allow_text_output=False)
    resolved = params.with_default_output_mode('native')
    assert resolved.output_mode == 'native'
    assert resolved.allow_text_output is True


def test_deferred_capability_ids_records_which_capabilities_defer_loading():
    """The field says which *capabilities* defer loading, which the tool definitions cannot.

    `capability_id` records which capability contributed a tool, and `defer_loading` is set both by
    a deferred capability and by a search-gated tool inside an always-on one — so ownership alone
    cannot tell the two apart. The agent reads it off the capability instances instead, and
    adapters and reveal-gating rely on it meaning what it says.
    """
    params = ModelRequestParameters(
        function_tools=[
            ToolDefinition(name='gated', defer_loading=True, capability_id='refunds'),
            ToolDefinition(name='searchable', defer_loading=True, capability_id='eager_capability'),
            ToolDefinition(name='plain', capability_id='eager_capability'),
        ],
        deferred_capability_ids={'refunds'},
    )
    assert params.deferred_capability_ids == {'refunds'}
    # `eager_capability` owns a deferred tool but does not itself defer, so it is absent — the
    # distinction that a derivation from the definitions would collapse.
    assert 'eager_capability' not in params.deferred_capability_ids


def test_old_serialized_payload_with_deferred_capability_ids_still_validates():
    """A v2.23-era Temporal payload carries the removed field; deserialization must not choke."""
    dumped = ta.dump_python(ModelRequestParameters(), mode='json')
    dumped['deferred_capability_ids'] = ['refunds']
    ta.validate_python(dumped)


def test_declared_tool_defs_never_drops_an_output_tool():
    """The visibility filter applies to function tools only.

    `tool_visibility` is keyed by name; a hidden function tool sharing a name with an output tool
    must not shadow the output tool out of the provider's `tools` collection.
    """
    output_tool = ToolDefinition(name='final_result', kind='output')
    params = ModelRequestParameters(
        function_tools=[ToolDefinition(name='final_result', defer_loading=True)],
        output_tools=[output_tool],
        tool_visibility={'final_result': 'withheld'},
    )
    assert params.declared_function_tools == []
    assert params.declared_tool_defs == {'final_result': output_tool}
