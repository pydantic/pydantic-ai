from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.coder._capability import _CoderGuidance  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_harness.coder._instructions import coder_instructions
from pydantic_ai_harness.coder._model_guidance import (
    MISREAD_CLASSIFY,
    PURE_FUNCTIONS,
    STRUCTURED_HANDOFF,
    ModelGuidance,
    is_openai_family,
    resolve_model_guidance,
)

pytestmark = pytest.mark.anyio

_MISREAD = 'If inputs seem inconsistent or goals impossible, assume you misread.'
_MISREAD_MAX_MARKER = 'A result that leaves a stated criterion unmet means your interpretation is likely wrong'


class _Named(TestModel):
    def __init__(self, system: str, name: str) -> None:
        super().__init__()
        self._system = system
        self._name = name

    @property
    def system(self) -> str:
        return self._system

    @property
    def model_name(self) -> str:
        return self._name


def _guidance(coder: Coder[Any]) -> _CoderGuidance[Any]:
    return next(c for c in coder.capabilities if isinstance(c, _CoderGuidance))


async def _run_guidance(coder: Coder[Any], system: str, name: str) -> str:
    run = cast(Any, await _guidance(coder).for_run(cast(Any, SimpleNamespace(model=_Named(system, name)))))
    return run.guidance


def test_resolve_prefers_model_glob_over_provider():
    provider_row = ModelGuidance('openai', add=('provider',))
    model_row = ModelGuidance('openai:gpt-6.2-*', add=('model',))
    table = (provider_row, model_row)
    assert resolve_model_guidance(table, _Named('openai', 'gpt-6.2-sol')) is model_row
    assert resolve_model_guidance(table, _Named('openai', 'gpt-5')) is provider_row
    assert resolve_model_guidance(table, _Named('anthropic', 'claude-opus-4')) is None


def test_resolve_matches_openai_family_on_gateways():
    row = ModelGuidance('openai', add=('family',))
    table = (row,)
    assert resolve_model_guidance(table, _Named('openai', 'gpt-6.2-sol')) is row
    assert resolve_model_guidance(table, _Named('openrouter', 'openai/gpt-6.2-sol')) is row
    assert resolve_model_guidance(table, _Named('azure', 'gpt-4o')) is row
    assert resolve_model_guidance(table, _Named('azure', 'o3-mini')) is row
    assert resolve_model_guidance(table, _Named('openrouter', 'anthropic/claude-opus-4')) is None
    assert resolve_model_guidance(table, _Named('anthropic', 'claude-opus-4')) is None


def test_resolve_model_glob_accepts_gateway_leaf_name():
    row = ModelGuidance('openai:gpt-6.2-*', add=('model',))
    assert resolve_model_guidance((row,), _Named('openrouter', 'openai/gpt-6.2-sol')) is row
    assert resolve_model_guidance((row,), _Named('openrouter', 'openai/gpt-5')) is None


def test_openai_family_is_narrow():
    assert is_openai_family(_Named('openai', 'gpt-6.2-sol'))
    assert is_openai_family(_Named('openrouter', 'openai/gpt-6.2-sol'))
    assert is_openai_family(_Named('azure', 'gpt-4o'))
    assert not is_openai_family(_Named('openrouter', 'anthropic/claude-opus-4'))
    assert not is_openai_family(_Named('groq', 'llama-3.1-70b'))
    assert not is_openai_family(_Named('ollama', 'olmo2'))


def test_resolve_unwraps_wrapper_models():
    row = ModelGuidance('openai')
    assert resolve_model_guidance((row,), WrapperModel(_Named('openai', 'gpt-6.2-sol'))) is row


def test_resolve_unwraps_fallback_models():
    row = ModelGuidance('openai')
    model = FallbackModel(_Named('openai', 'gpt-6.2-sol'), _Named('anthropic', 'claude-opus-4'))
    assert resolve_model_guidance((row,), model) is row
    other = FallbackModel(_Named('anthropic', 'claude-opus-4'), _Named('openai', 'gpt-6.2-sol'))
    assert resolve_model_guidance((row,), other) is None


def test_row_replaces_max_misread_and_adds_sentences():
    base, base_delegate = coder_instructions(autonomy='max', delegate_tool=None)
    row = ModelGuidance('openai', replace={'misread': MISREAD_CLASSIFY}, add=(STRUCTURED_HANDOFF, PURE_FUNCTIONS))
    text, delegate = coder_instructions(autonomy='max', delegate_tool=None, model_guidance=row)
    for before, after in ((base, text), (base_delegate, delegate)):
        assert _MISREAD_MAX_MARKER in before and _MISREAD_MAX_MARKER not in after
        assert MISREAD_CLASSIFY in after and STRUCTURED_HANDOFF in after and PURE_FUNCTIONS in after
        assert after.endswith(before.rsplit('\n\n', 1)[-1])


def test_empty_replacement_drops_fragment():
    text, _ = coder_instructions(
        autonomy='max', delegate_tool=None, model_guidance=ModelGuidance('openai', replace={'misread': ''})
    )
    assert 'assume you misread' not in text
    assert _MISREAD_MAX_MARKER not in text


async def test_default_table_applies_only_for_openai_family_at_max():
    high = _guidance(Coder[Any](autonomy='high'))
    openai_high = await _run_guidance(Coder[Any](autonomy='high'), 'openai', 'gpt-6.2-sol')
    assert openai_high == high.guidance
    assert MISREAD_CLASSIFY not in openai_high
    assert STRUCTURED_HANDOFF not in openai_high
    assert PURE_FUNCTIONS not in openai_high
    assert _MISREAD in openai_high

    max_coder = Coder[Any](autonomy='max')
    openai_max = await _run_guidance(max_coder, 'openai', 'gpt-6.2-sol')
    assert openai_max != _guidance(max_coder).guidance
    assert MISREAD_CLASSIFY in openai_max
    assert STRUCTURED_HANDOFF in openai_max
    assert PURE_FUNCTIONS in openai_max
    assert _MISREAD_MAX_MARKER not in openai_max

    anthropic_max = await _run_guidance(max_coder, 'anthropic', 'claude-opus-4')
    assert anthropic_max == _guidance(max_coder).guidance
    assert PURE_FUNCTIONS not in anthropic_max
    assert _MISREAD_MAX_MARKER in anthropic_max

    gateway_max = await _run_guidance(max_coder, 'openrouter', 'openai/gpt-6.2-sol')
    assert PURE_FUNCTIONS in gateway_max
    routed_other = await _run_guidance(max_coder, 'openrouter', 'anthropic/claude-opus-4')
    assert routed_other == _guidance(max_coder).guidance


async def test_empty_table_disables_default_guidance():
    off = Coder[Any](autonomy='max', model_guidance=())
    text = await _run_guidance(off, 'openai', 'gpt-6.2-sol')
    assert text == _guidance(off).guidance
    assert MISREAD_CLASSIFY not in text
    assert PURE_FUNCTIONS not in text
    assert _MISREAD_MAX_MARKER in text
