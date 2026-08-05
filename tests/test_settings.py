import importlib
import pkgutil

import pytest

from pydantic_ai import Agent, models
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings, merge_model_settings

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


_MODEL_MODULE_NAMES = [module_info.name for module_info in pkgutil.iter_modules(models.__path__, f'{models.__name__}.')]


def _discover_model_settings() -> tuple[dict[str, type], list[str]]:
    """Collect every `ModelSettings` subclass defined by a `pydantic_ai.models` submodule.

    Derived from the package rather than a hardcoded list so a new provider is covered the moment it
    lands, and a renamed module can't silently drop a settings class from the prefix check.
    """
    settings_classes: dict[str, type] = {}
    unimportable: list[str] = []
    for module_name in _MODEL_MODULE_NAMES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:  # pragma: lax no cover
            unimportable.append(f'{module_name} ({e})')
            continue
        for name, obj in vars(module).items():
            if not isinstance(obj, type) or obj.__module__ != module_name:
                continue
            # `TypedDict` subclasses report `dict` as their only `__bases__`; `__orig_bases__` keeps the real chain.
            bases = list(getattr(obj, '__orig_bases__', ()))
            while bases:
                base = bases.pop()
                if base is ModelSettings:
                    settings_classes[name] = obj
                    break
                bases.extend(getattr(base, '__orig_bases__', ()))
    return settings_classes, unimportable


_MODEL_SETTINGS_CLASSES, _UNIMPORTABLE_MODEL_MODULES = _discover_model_settings()

# Provider-specific settings fields are namespaced with the provider's name, which is also the name of
# the module the provider lives in. `mcp_sampling` is not a provider integration but the MCP sampling
# pseudo-model, so its public fields are namespaced after the protocol instead.
_PREFIX_OVERRIDES = {'mcp_sampling': 'mcp_'}


@pytest.mark.parametrize('settings_cls', _MODEL_SETTINGS_CLASSES.values(), ids=list(_MODEL_SETTINGS_CLASSES))
def test_specific_prefix_settings(settings_cls: type):
    module_name = settings_cls.__module__.rsplit('.', maxsplit=1)[-1]
    prefix = _PREFIX_OVERRIDES.get(module_name, f'{module_name}_')
    global_settings = set(ModelSettings.__annotations__.keys())
    specific_settings = set(settings_cls.__annotations__.keys()) - global_settings
    assert all(setting.startswith(prefix) for setting in specific_settings), (
        f'{prefix} is not a prefix for {specific_settings}'
    )


def test_model_settings_discovery():
    # The number of settings classes depends on which optional groups are installed, so the rot guard
    # is on the module walk instead: if that quietly returns (almost) nothing because the package moved,
    # every prefix check above silently disappears, which is what the hardcoded provider list did.
    assert len(_MODEL_MODULE_NAMES) >= 15, f'only walked {_MODEL_MODULE_NAMES}'
    assert _MODEL_SETTINGS_CLASSES, f'no settings classes found, unimportable modules: {_UNIMPORTABLE_MODEL_MODULES}'


@pytest.mark.parametrize(
    'model', ['openai', 'anthropic', 'bedrock', 'mistral', 'groq', 'cohere', 'google'], indirect=True
)
async def test_stop_settings(allow_model_requests: None, model: Model) -> None:
    agent = Agent(model=model, model_settings=ModelSettings(stop_sequences=['Paris']))
    result = await agent.run(
        'What is the capital of France? Give me an answer that contains the word "Paris", but is not the first word.'
    )

    # NOTE: Bedrock has a slightly different behavior. It will include the stop sequence in the response.
    if model.system == 'bedrock':
        assert result.output.endswith('Paris')
    else:
        assert 'Paris' not in result.output


class TestMergeModelSettingsThinking:
    """merge_model_settings with unified thinking fields."""

    def test_merge_thinking_bool_override(self):
        base: ModelSettings = {'thinking': True}
        overrides: ModelSettings = {'thinking': False}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') is False

    def test_merge_effort_override(self):
        base: ModelSettings = {'thinking': 'low'}
        overrides: ModelSettings = {'thinking': 'high'}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') == 'high'

    def test_merge_preserves_non_thinking_settings(self):
        base: ModelSettings = {'max_tokens': 1000, 'temperature': 0.5}
        overrides: ModelSettings = {'thinking': True}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('max_tokens') == 1000
        assert result.get('temperature') == 0.5
        assert result.get('thinking') is True

    def test_merge_with_none_returns_base(self):
        base: ModelSettings = {'thinking': True}
        result = merge_model_settings(base, None)
        assert result == base

    def test_merge_with_none_base_returns_overrides(self):
        overrides: ModelSettings = {'thinking': True}
        result = merge_model_settings(None, overrides)
        assert result == overrides

    def test_merge_with_both_none(self):
        result = merge_model_settings(None, None)
        assert result is None


class TestMergeModelSettingsServiceTier:
    """merge_model_settings with unified service_tier field."""

    def test_merge_service_tier_override(self):
        base: ModelSettings = {'service_tier': 'default'}
        overrides: ModelSettings = {'service_tier': 'priority'}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('service_tier') == 'priority'

    def test_merge_preserves_non_service_tier_settings(self):
        base: ModelSettings = {'max_tokens': 1000, 'temperature': 0.5}
        overrides: ModelSettings = {'service_tier': 'flex'}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('max_tokens') == 1000
        assert result.get('temperature') == 0.5
        assert result.get('service_tier') == 'flex'
