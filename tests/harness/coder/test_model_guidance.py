from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai_harness.coder import (
    Autonomy,
    Coder,
    ModelGuidance,
    is_openai_family,
    resolve_model_guidance,
)

_MISREAD = (
    'If inputs seem inconsistent or goals impossible, assume you misread. '
    'Re-read, and try other interpretations before settling.'
)
_MISREAD_MAX = (
    'If inputs seem inconsistent or a criterion impossible (no feasible plan, matching item, or reachable target), '
    'assume you misread. Re-read the task and inputs, and try interpretations meeting every stated criterion '
    'before settling for a partial answer. A result that leaves a stated criterion unmet means your '
    'interpretation is likely wrong, not that the answer is "none".'
)
_MISREAD_CLASSIFY = (
    'If inputs look inconsistent, first check whether the task or its rules classify them '
    '(for example as malformed, conflicting, or rejected) and apply that rule. '
    'Only when no rule covers them, assume you misread: re-read, and try other interpretations before settling.'
)
_STRUCTURED_HANDOFF = (
    'When asked for notes or a handoff, use headed sections: each decision and why, '
    'the exact verification commands with their observed results, the remaining risk, '
    'and what was out of scope or not done.'
)
_PURE_FUNCTIONS = (
    'When a task says a function must derive its answer only from its arguments '
    '(no files, commands, network, or mutation), write it as plain self-contained Python: dicts, lists, loops, '
    'and helper functions. Import only the basic standard modules it needs (re, math, json, copy, heapq, '
    'bisect, datetime); no typing, collections.abc, __future__, or dataclasses imports. '
    "Don't define classes or generators (yield), and don't call iter(), id(), divmod(), hasattr(), or object()."
)


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


class _NamedFunction(FunctionModel):
    def __init__(self, system: str, name: str, seen: list[str]) -> None:
        def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions or '')
            return ModelResponse(parts=[TextPart(content='done')])

        super().__init__(respond, model_name=name)
        self._system = system


async def _run_guidance(
    tmp_path: Path,
    *,
    system: str,
    name: str,
    autonomy: Autonomy = 'max',
    model_guidance: Sequence[ModelGuidance] | None = None,
) -> str:
    seen: list[str] = []
    agent = Agent[object](
        _NamedFunction(system, name, seen),
        capabilities=[
            LocalWorkspace(tmp_path),
            Coder[object](autonomy=autonomy, repo_context=False, model_guidance=model_guidance),
        ],
    )
    await agent.run('go')
    return seen[0]


def test_resolve_prefers_model_glob_over_provider():
    """Pure name/provider lookup; this issues no model-provider request to record."""
    provider_row = ModelGuidance('openai', add=('provider',))
    model_row = ModelGuidance('openai:gpt-6.2-*', add=('model',))
    table = (provider_row, model_row)
    assert resolve_model_guidance(table, _Named('openai', 'gpt-6.2-sol')) is model_row
    assert resolve_model_guidance(table, _Named('openai', 'gpt-5')) is provider_row
    assert resolve_model_guidance(table, _Named('anthropic', 'claude-opus-4')) is None


def test_resolve_matches_openai_family_on_gateways():
    """Pure name/provider lookup; this issues no model-provider request to record."""
    row = ModelGuidance('openai', add=('family',))
    table = (row,)
    assert resolve_model_guidance(table, _Named('openai', 'gpt-6.2-sol')) is row
    assert resolve_model_guidance(table, _Named('openrouter', 'openai/gpt-6.2-sol')) is row
    assert resolve_model_guidance(table, _Named('azure', 'gpt-4o')) is row
    assert resolve_model_guidance(table, _Named('azure', 'o3-mini')) is row
    assert resolve_model_guidance(table, _Named('azure', 'chatgpt-4o-latest')) is row
    assert resolve_model_guidance(table, _Named('openrouter', 'anthropic/claude-opus-4')) is None
    assert resolve_model_guidance(table, _Named('anthropic', 'claude-opus-4')) is None


def test_resolve_model_glob_accepts_gateway_leaf_name():
    """Pure name/provider lookup; this issues no model-provider request to record."""
    row = ModelGuidance('openai:gpt-6.2-*', add=('model',))
    assert resolve_model_guidance((row,), _Named('openrouter', 'openai/gpt-6.2-sol')) is row
    assert resolve_model_guidance((row,), _Named('openrouter', 'openai/gpt-5')) is None


def test_openai_family_is_narrow():
    """Pure name/provider lookup; this issues no model-provider request to record."""
    assert is_openai_family(_Named('openai', 'gpt-6.2-sol'))
    assert is_openai_family(_Named('openrouter', 'openai/gpt-6.2-sol'))
    assert is_openai_family(_Named('azure', 'gpt-4o'))
    assert is_openai_family(_Named('azure', 'chatgpt-4o-latest'))
    assert is_openai_family(_Named('azure', 'o3-mini'))
    assert not is_openai_family(_Named('openrouter', 'anthropic/claude-opus-4'))
    assert not is_openai_family(_Named('groq', 'llama-3.1-70b'))
    assert not is_openai_family(_Named('ollama', 'olmo2'))


def test_resolve_unwraps_wrapper_models():
    """Pure name/provider lookup; this issues no model-provider request to record."""
    row = ModelGuidance('openai')
    assert resolve_model_guidance((row,), WrapperModel(_Named('openai', 'gpt-6.2-sol'))) is row


def test_resolve_unwraps_fallback_models():
    """Pure name/provider lookup; this issues no model-provider request to record."""
    row = ModelGuidance('openai')
    model = FallbackModel(_Named('openai', 'gpt-6.2-sol'), _Named('anthropic', 'claude-opus-4'))
    assert resolve_model_guidance((row,), model) is row
    other = FallbackModel(_Named('anthropic', 'claude-opus-4'), _Named('openai', 'gpt-6.2-sol'))
    assert resolve_model_guidance((row,), other) is None


async def test_default_table_applies_only_for_openai_family_at_max(tmp_path: Path):
    openai_high = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='high')
    assert _MISREAD in openai_high
    assert _MISREAD_CLASSIFY not in openai_high
    assert _STRUCTURED_HANDOFF not in openai_high
    assert _PURE_FUNCTIONS not in openai_high

    openai_max = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='max')
    assert _MISREAD_CLASSIFY in openai_max
    assert _STRUCTURED_HANDOFF in openai_max
    assert _PURE_FUNCTIONS in openai_max
    assert _MISREAD_MAX not in openai_max

    anthropic_max = await _run_guidance(tmp_path, system='anthropic', name='claude-opus-4', autonomy='max')
    assert anthropic_max != openai_max
    assert _PURE_FUNCTIONS not in anthropic_max
    assert _MISREAD_MAX in anthropic_max

    gateway_max = await _run_guidance(tmp_path, system='openrouter', name='openai/gpt-6.2-sol', autonomy='max')
    assert _PURE_FUNCTIONS in gateway_max
    routed_other = await _run_guidance(tmp_path, system='openrouter', name='anthropic/claude-opus-4', autonomy='max')
    assert _PURE_FUNCTIONS not in routed_other
    assert _MISREAD_MAX in routed_other


async def test_empty_table_disables_default_guidance(tmp_path: Path):
    text = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='max', model_guidance=())
    assert _MISREAD_CLASSIFY not in text
    assert _PURE_FUNCTIONS not in text
    assert _MISREAD_MAX in text


async def test_explicit_table_applies_at_high_autonomy(tmp_path: Path):
    row = ModelGuidance('openai', replace={'misread': _MISREAD_CLASSIFY}, add=(_STRUCTURED_HANDOFF, _PURE_FUNCTIONS))
    text = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='high', model_guidance=(row,))
    assert _MISREAD_CLASSIFY in text
    assert _STRUCTURED_HANDOFF in text
    assert _PURE_FUNCTIONS in text
    assert _MISREAD not in text


async def test_empty_replacement_drops_max_misread_fragment(tmp_path: Path):
    text = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        autonomy='max',
        model_guidance=(ModelGuidance('openai', replace={'misread': ''}),),
    )
    assert 'assume you misread' not in text
    assert _MISREAD_MAX not in text
