"""Deprecated profile key spellings translate to their current ones with a warning."""

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


@pytest.mark.parametrize(
    ('legacy_key', 'key'),
    [
        ('openai_supports_tool_choice_required', 'supports_forced_tool_choice'),
        ('grok_supports_tool_choice_required', 'supports_forced_tool_choice'),
        ('anthropic_supports_forced_tool_choice', 'supports_forced_tool_choice'),
        ('openai_supports_forced_tool_choice_with_thinking', 'supports_forced_tool_choice_with_thinking'),
        ('openrouter_supports_forced_tool_choice_with_thinking', 'supports_forced_tool_choice_with_thinking'),
        ('openai_reasoning_enabled_by_default', 'thinking_enabled_by_default'),
    ],
)
def test_legacy_provider_keys_translate_to_model_profile_keys(legacy_key: str, key: str):
    """Provider-prefixed tool-forcing and thinking keys moved to `ModelProfile`, so every model family shares them."""
    with pytest.warns(PydanticAIDeprecationWarning, match=f'`{legacy_key}` is deprecated, use `{key}` instead'):
        assert merge_profile(ModelProfile(), {legacy_key: False}) == {key: False}  # pyright: ignore[reportArgumentType]
    with pytest.warns(PydanticAIDeprecationWarning):
        model = TestModel(profile={legacy_key: False})  # pyright: ignore[reportArgumentType]
        assert model.profile.get(key) is False
        assert legacy_key not in model.profile


def test_two_legacy_spellings_of_a_capability_must_both_allow_it():
    with pytest.warns(PydanticAIDeprecationWarning):
        profile = merge_profile(
            {'openai_supports_tool_choice_required': True, 'anthropic_supports_forced_tool_choice': False}  # pyright: ignore[reportArgumentType]
        )
    assert profile == {'supports_forced_tool_choice': False}


def test_current_provider_key_spelling_wins_over_legacy_in_the_same_profile():
    with pytest.warns(PydanticAIDeprecationWarning):
        profile = merge_profile(
            {'openai_supports_tool_choice_required': False, 'supports_forced_tool_choice': True}  # pyright: ignore[reportArgumentType]
        )
    assert profile == {'supports_forced_tool_choice': True}


def test_legacy_provider_key_from_callable_profile_wins_over_carried_over_default():
    """A callable profile starts from the resolved profile, which already carries the current spelling's default,
    so a legacy key it adds has to win over that carried-over value."""
    with pytest.warns(PydanticAIDeprecationWarning, match='`openai_supports_tool_choice_required` is deprecated'):
        model = TestModel(profile=lambda profile: {**profile, 'openai_supports_tool_choice_required': False})  # pyright: ignore[reportArgumentType]
        assert model.profile.get('supports_forced_tool_choice') is False
