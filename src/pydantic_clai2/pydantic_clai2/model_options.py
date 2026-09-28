"""Model-aware editor choices. Protocol behavior remains in Pydantic AI profiles."""

import re

from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.openai import openai_model_profile

from .model_settings import ModelSettingsForm, model_defaults


def model_options(*, model: str) -> dict[str, tuple[str, ...]]:
    """Offer native controls only for the provider that will consume them."""
    provider, _, name = model.partition(':')
    name = name.rsplit('/', 1)[-1]
    family_defaults = bool(model_defaults(model=model))
    options: dict[str, tuple[str, ...]] = {key: () for key in ('max_tokens', 'temperature', 'seed', 'custom_params')}
    if provider in ('openai', 'openai-chat', 'openai-responses', 'openai-codex'):
        options = _openai_options(provider=provider, name=name, family_defaults=family_defaults)
    elif provider == 'anthropic':
        options = _anthropic_options(name=name)
    elif provider in ('google', 'google-gla', 'google-vertex'):
        options['top_p'] = ()
        if name.startswith(('gemini-2.5', 'gemini-3')):
            options['thinking'] = ()
    elif family_defaults or openai_model_profile(name).get('openai_supports_reasoning', False):
        options.pop('temperature')
        options.pop('seed')
        options['thinking'] = ()
        if provider in ('openrouter', 'vllm'):
            options = _openai_options(provider='openai-chat', name=name, family_defaults=family_defaults)
    glm = _glm_version(name=name)
    if glm is not None and glm >= (4, 5):
        options['glm_thinking'] = ()
        options['glm_clear_thinking'] = ()
        if glm >= (5, 2):
            options['glm_reasoning_effort'] = ()
    return options


def _anthropic_options(*, name: str) -> dict[str, tuple[str, ...]]:
    options: dict[str, tuple[str, ...]] = {key: () for key in ('max_tokens', 'temperature', 'top_p', 'custom_params')}
    claude = anthropic_model_profile(name) or {}
    adaptive = claude.get('anthropic_supports_adaptive_thinking', False)
    options['anthropic_thinking_mode'] = ('adaptive', 'disabled') if adaptive else ('enabled', 'disabled')
    if not adaptive:
        options['anthropic_thinking_budget'] = ()
    if claude.get('anthropic_supports_effort', False):
        options['anthropic_effort'] = _anthropic_efforts(name=name)
    if adaptive:
        options['anthropic_preserved_thinking'] = ()
    if any(tag in name.lower() for tag in _FABLE_5_1_TAGS):
        options['anthropic_thinking_display'] = ('updates', 'summarized')
    if not adaptive and re.match(r'claude-(?:opus|sonnet|haiku)-4', name.lower()):
        options['anthropic_interleaved_thinking'] = ()
    if claude.get('anthropic_disallows_sampling_settings', False):
        for key in ('temperature', 'top_p', 'top_k'):
            options.pop(key, None)
    return options


def _openai_options(*, provider: str, name: str, family_defaults: bool) -> dict[str, tuple[str, ...]]:
    options: dict[str, tuple[str, ...]] = {key: () for key in ('max_tokens', 'temperature', 'seed', 'custom_params')}
    options['top_p'] = ()
    options['service_tier'] = ()
    if provider != 'openai-chat':
        options.pop('seed')
    profile = openai_model_profile(name)
    if family_defaults or profile.get('openai_supports_reasoning', False):
        for key in ('temperature', 'top_p', 'seed'):
            options.pop(key, None)
        options['thinking'] = ()
        efforts = _openai_efforts(name=name)
        options['openai_reasoning_effort'] = tuple(efforts)
        if provider != 'openai-chat':
            options['openai_reasoning_summary'] = ()
            options['openai_reasoning_context'] = (
                ('auto', 'current_turn', 'all_turns')
                if family_defaults or profile.get('openai_responses_supports_reasoning_context', False)
                else ('auto', 'current_turn')
            )
            if family_defaults or profile.get('openai_responses_supports_reasoning_mode', False):
                options['openai_reasoning_mode'] = ()
            options['openai_text_verbosity'] = ()
    return options


def validate_model_options(*, model: str, form: ModelSettingsForm) -> None:
    """Reject unavailable native settings and incompatible thinking budgets before saving."""
    options = model_options(model=model)
    for key, value in form.model_dump(exclude_none=True).items():
        if key not in options:
            raise ValueError(f'{key} is not supported for {model}. Reset this override first.')
        choices = options[key]
        if choices and str(value) not in choices:
            raise ValueError(f'{key}: choose {", ".join(choices)}.')
    if form.anthropic_thinking_budget is not None and form.anthropic_thinking_mode != 'enabled':
        raise ValueError('Set anthropic_thinking_mode to enabled before setting its budget.')
    if form.anthropic_thinking_mode == 'enabled':
        if form.max_tokens is not None and (form.anthropic_thinking_budget or 10000) >= form.max_tokens:
            raise ValueError('Thinking budget must be less than max_tokens.')
        if form.temperature not in (None, 1.0) or form.top_p is not None:
            raise ValueError('Classic thinking requires temperature=1 and no top_p override.')
    if model.startswith('anthropic:'):
        profile = anthropic_model_profile(model.partition(':')[2]) or {}
        if (
            profile.get('anthropic_disallows_top_effort_when_thinking_disabled', False)
            and form.anthropic_thinking_mode == 'disabled'
            and form.anthropic_effort in ('xhigh', 'max')
        ):
            raise ValueError('This model requires thinking for xhigh or max effort.')
    if form.anthropic_thinking_mode == 'disabled' and (
        form.anthropic_thinking_display is not None or form.anthropic_preserved_thinking is not None
    ):
        raise ValueError('Set extended thinking to enabled or adaptive before using its display or binding.')
    if form.glm_thinking == 'disabled' and form.glm_reasoning_effort is not None:
        raise ValueError('Set GLM thinking to enabled before setting its reasoning effort.')


# Fable 5.1 surfaces its inter-tool progress updates; both spellings appear in the wild.
_FABLE_5_1_TAGS: tuple[str, ...] = ('fable-5-1', '5-1-fable', 'fable-5.1', '5.1-fable')


def _glm_version(*, name: str) -> tuple[int, int] | None:
    """The GLM generation in a name, as a pair so 4.10 sorts above 4.5."""
    match = re.search(r'\bglm-(\d+)(?:\.(\d+))?', name.lower())
    return (int(match[1]), int(match[2] or 0)) if match else None


def _openai_efforts(*, name: str) -> tuple[str, ...]:
    profile = openai_model_profile(name)
    if 'chat' in name:
        return ('medium',)
    efforts = ['low', 'medium', 'high']
    if profile.get('openai_supports_reasoning_effort_none', False):
        efforts.insert(0, 'none')
    version = re.search(r'gpt-(\d+)(?:\.(\d+))?', name)
    if version:
        number = (int(version[1]), int(version[2] or 0))
        if number == (5, 0):
            efforts.insert(0, 'minimal')
        if number >= (5, 2) or 'codex-max' in name:
            efforts.append('xhigh')
        if number >= (5, 6):
            efforts.append('max')
    return tuple(efforts)


def _anthropic_efforts(*, name: str) -> tuple[str, ...]:
    profile = anthropic_model_profile(name) or {}
    efforts = ['low', 'medium', 'high']
    if profile.get('anthropic_supports_xhigh_effort', False):
        efforts.append('xhigh')
    # Max effort starts with Opus 4.6; Sonnet only offers low/medium/high.
    if profile.get('anthropic_supports_adaptive_thinking', False) and 'sonnet' not in name:
        efforts.append('max')
    return tuple(efforts)
