"""The v2.23-released spellings of the reveal-channel profile keys translate with a warning."""

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.models import ModelRequestParameters, ToolDefinition
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile, merge_profile


def test_legacy_tool_additions_translates_and_warns():
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`tool_additions` is deprecated'):
        profile = merge_profile({'tool_additions': 'by_reference'})
    assert profile == {'tool_addition_mode': 'by_reference'}


def test_legacy_deferred_tools_require_tool_search_translates_and_warns():
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`deferred_tools_require_tool_search` is deprecated'):
        assert merge_profile({'deferred_tools_require_tool_search': True}) == {'tool_deferral_mode': 'with_tool_search'}
    # `False` carried no signal on its own — deferral capability came from native tool-search
    # support — so it is dropped rather than translated to a mode it never meant.
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`deferred_tools_require_tool_search` is deprecated'):
        assert merge_profile({'deferred_tools_require_tool_search': False}) == {}


def test_current_spelling_wins_over_legacy_in_the_same_profile():
    with pytest.warns(PydanticAIDeprecationWarning):
        profile = merge_profile({'tool_additions': 'by_reference', 'tool_addition_mode': 'with_definitions'})
    assert profile == {'tool_addition_mode': 'with_definitions'}


def test_legacy_key_in_override_still_overrides_base():
    with pytest.warns(PydanticAIDeprecationWarning):
        profile = merge_profile({'tool_addition_mode': 'with_definitions'}, {'tool_additions': 'by_reference'})
    assert profile == {'tool_addition_mode': 'by_reference'}


def test_legacy_keys_reach_resolution_through_model_profile_argument():
    """The whole point: a v2.23-era `Model(profile=...)` keeps driving the resolve table."""
    with pytest.warns(PydanticAIDeprecationWarning):
        model = TestModel(profile=ModelProfile(tool_additions='by_reference', deferred_tools_require_tool_search=True))
        assert model.tool_addition_mode == 'by_reference'
        assert model.tool_deferral_mode == 'with_tool_search'

    # The callable form bypasses `merge_profile`, so resolution translates its result directly.
    with pytest.warns(PydanticAIDeprecationWarning):
        callable_model = TestModel(profile=lambda _default: ModelProfile(tool_additions='with_definitions'))
        assert callable_model.tool_addition_mode == 'with_definitions'

    # `Model.profile` is cached per instance, so the warning fires once at resolution, not on
    # every request — this call must run silently against the already-translated profile. And the
    # translated claim drives the resolve table with v2.23 semantics: `with_tool_search` grants no
    # deferral in a request that sends no tool-search tool, so the hidden tool stays withheld.
    hidden = ToolDefinition(name='hidden_tool', defer_loading=True, capability_id='refunds')
    _, prepared = model.prepare_request(None, ModelRequestParameters(function_tools=[hidden]))
    assert prepared.tool_visibility == {'hidden_tool': 'withheld'}
