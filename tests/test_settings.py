import ast
import importlib
import pkgutil
import re
from pathlib import Path

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


# The label each `Supported by:` list uses for a provider, mapped to the `pydantic_ai.models` module that
# implements it. Several fields say `Gemini` where the module is `google`, and `Z.AI` where it's `zai`.
_SUPPORTED_BY_LABELS = {
    'Anthropic': 'anthropic',
    'Bedrock': 'bedrock',
    'Cerebras': 'cerebras',
    'Cohere': 'cohere',
    'Gemini': 'google',
    'Google': 'google',
    'Groq': 'groq',
    'Hugging Face': 'huggingface',
    'MCP Sampling': 'mcp_sampling',
    'Mistral': 'mistral',
    'OpenAI': 'openai',
    'OpenRouter': 'openrouter',
    'xAI': 'xai',
    'Z.AI': 'zai',
}

# Fields a provider never reads as a `model_settings` lookup, so the source scan below can't see them:
# `tool_choice` is resolved by `models._tool_choice.resolve_tool_choice` and `thinking` by
# `ModelRequestParameters.thinking`, both of which every model receives whether it honours it or not.
# Their `Supported by:` lists are therefore not machine-checkable and are excluded here.
_INDIRECTLY_CONSUMED_FIELDS = {'tool_choice', 'thinking'}

_MODEL_SOURCES = {
    (path := Path(models.__file__).parent / f'{module_name.rsplit(".", maxsplit=1)[-1]}.py').stem: path.read_text(
        encoding='utf-8'
    )
    for module_name in _MODEL_MODULE_NAMES
}


def _supported_by(field_docstring: str) -> list[str]:
    """The provider labels a field docstring's `Supported by:` list names.

    Returns an empty list when the field has no such list, which the test below rejects: a general
    setting whose provider support isn't documented anywhere is the same documentation gap this test
    exists to catch.
    """
    match = re.search(r'Supported by:\n\n(.*?)(?:\n\n|\Z)', field_docstring, re.DOTALL)
    # Entries may carry a parenthesized caveat, e.g. `* OpenAI (some models, not o1)`.
    return (
        [
            re.sub(r'\s*\(.*', '', line.strip().removeprefix('* '))
            for line in match.group(1).splitlines()
            if line.strip().startswith('* ')
        ]
        if match
        else []
    )


def _documented_support() -> dict[str, list[str]]:
    """Map each `ModelSettings` field to the provider labels its `Supported by:` list names."""
    settings_source = Path(models.__file__).parent.parent / 'settings.py'
    class_def = next(
        node
        for node in ast.parse(settings_source.read_text(encoding='utf-8')).body
        if isinstance(node, ast.ClassDef) and node.name == 'ModelSettings'
    )
    fields = [
        node.target.id
        for node in class_def.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    docstrings = {
        annotation.target.id: ast.literal_eval(following.value)
        for annotation, following in zip(class_def.body, class_def.body[1:])
        if isinstance(annotation, ast.AnnAssign)
        and isinstance(annotation.target, ast.Name)
        and isinstance(following, ast.Expr)
        and isinstance(following.value, ast.Constant)
        and isinstance(following.value.value, str)
    }
    return {field: _supported_by(docstrings.get(field, '')) for field in fields}


_DOCUMENTED_SUPPORT = _documented_support()


def _reads(field: str) -> set[str]:
    """The `pydantic_ai.models` modules that forward `field` to their provider.

    Scanned from the source rather than by importing, so an uninstalled optional group can't silently
    drop a module from the comparison the way it would if the import were allowed to fail.
    """
    # `OpenAIResponsesModel` lives in `openai.py`, so its reads are attributed to `openai` either way.
    read_by = {
        name
        for name, source in _MODEL_SOURCES.items()
        if re.search(rf"""(model_settings|settings)(\.get\(\s*|\[\s*)['"]{field}['"]""", source)
    }
    # `xai.py` forwards settings through `_XAI_MODEL_SETTINGS_MAPPING` instead of looking each one up.
    if re.search(rf"""^\s*['"]{field}['"]:""", _MODEL_SOURCES['xai'], re.MULTILINE):
        read_by.add('xai')
    return read_by


@pytest.mark.parametrize('field', sorted(_DOCUMENTED_SUPPORT.keys() - _INDIRECTLY_CONSUMED_FIELDS))
def test_supported_by_matches_implementation(field: str):
    """Every `Supported by:` list names exactly the provider modules that read that field.

    These lists are the only place a user can find out whether a general setting reaches a given
    provider, and a setting that doesn't is silently dropped rather than rejected — so a stale list
    is indistinguishable from a broken provider. Twelve had drifted: `huggingface.py` reads nine
    general settings and was named by none of them, `mistral.py` reads `parallel_tool_calls`, three
    models merge into `extra_body`, and xAI's `_XAI_MODEL_SETTINGS_MAPPING` covers neither `timeout`
    nor `extra_headers` though both claimed it.

    A unit test rather than a VCR one: it asserts a property of the source, which no recorded
    interaction can express.
    """
    labels = _DOCUMENTED_SUPPORT[field]
    assert labels, 'no `Supported by:` list, so there is nowhere for a user to look up this setting'

    unknown = set(labels) - _SUPPORTED_BY_LABELS.keys()
    assert not unknown, f'unrecognised provider name(s) {unknown}, add them to `_SUPPORTED_BY_LABELS`'

    assert _reads(field) == {_SUPPORTED_BY_LABELS[label] for label in labels}


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
