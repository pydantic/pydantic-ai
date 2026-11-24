from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai_harness.coder import MODEL_GUIDANCE, Autonomy, Coder, ModelGuidance

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
_ENVIRONMENT_CONSTRAINTS = (
    'When a task states constraints on where or how the code will run '
    '(a restricted or sandboxed interpreter, no dependencies, a specific Python version, pure functions, no I/O), '
    "find out what's actually allowed before writing code. Then prefer the simplest constructs that fit, "
    'and if possible check the code under those constraints.'
)


class _SystemFunction(FunctionModel):
    """A function model that reports the given provider, as `Model.system`."""

    def __init__(self, function: FunctionDef, *, system: str, name: str) -> None:
        super().__init__(function, model_name=name)
        self._provider_system = system

    @property
    def system(self) -> str:
        return self._provider_system


class _NamedFunction(_SystemFunction):
    def __init__(self, system: str, name: str, seen: list[str]) -> None:
        def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info.instructions or '')
            return ModelResponse(parts=[TextPart(content='done')])

        super().__init__(respond, system=system, name=name)


async def _run_guidance(
    tmp_path: Path,
    *,
    system: str,
    name: str,
    autonomy: Autonomy = 'max',
    model_guidance: Sequence[ModelGuidance] | None = None,
    wrap: Callable[[FunctionModel], FunctionModel | WrapperModel | FallbackModel] | None = None,
) -> str:
    """Run Coder once with a local function model and return the instructions it received.

    The lookup is local: the function model issues no provider request to record.
    """
    seen: list[str] = []
    inner = _NamedFunction(system, name, seen)
    agent = Agent[object](
        wrap(inner) if wrap is not None else inner,
        capabilities=[
            LocalWorkspace(tmp_path),
            Coder[object](autonomy=autonomy, repo_context=False, model_guidance=model_guidance),
        ],
    )
    await agent.run('go')
    return seen[0]


async def test_default_table_applies_only_for_openai_family_at_max(tmp_path: Path):
    """Default adjustments are for OpenAI-family models at max autonomy only; no provider request."""
    openai_high = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='high')
    assert _MISREAD in openai_high
    assert _MISREAD_CLASSIFY not in openai_high
    assert _STRUCTURED_HANDOFF not in openai_high
    assert _ENVIRONMENT_CONSTRAINTS not in openai_high

    openai_max = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='max')
    assert _MISREAD_CLASSIFY in openai_max
    assert _STRUCTURED_HANDOFF in openai_max
    assert _ENVIRONMENT_CONSTRAINTS in openai_max
    assert _MISREAD_MAX not in openai_max

    anthropic_max = await _run_guidance(tmp_path, system='anthropic', name='claude-opus-4', autonomy='max')
    assert anthropic_max != openai_max
    assert _ENVIRONMENT_CONSTRAINTS not in anthropic_max
    assert _MISREAD_MAX in anthropic_max

    gateway_max = await _run_guidance(tmp_path, system='openrouter', name='openai/gpt-6.2-sol', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS in gateway_max
    routed_other = await _run_guidance(tmp_path, system='openrouter', name='anthropic/claude-opus-4', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS not in routed_other
    assert _MISREAD_MAX in routed_other

    azure_gpt = await _run_guidance(tmp_path, system='azure', name='gpt-4o', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS in azure_gpt
    azure_chatgpt = await _run_guidance(tmp_path, system='azure', name='chatgpt-4o-latest', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS in azure_chatgpt
    azure_o = await _run_guidance(tmp_path, system='azure', name='o3-mini', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS in azure_o
    groq = await _run_guidance(tmp_path, system='groq', name='llama-3.1-70b', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS not in groq
    oss = await _run_guidance(tmp_path, system='groq', name='gpt-oss-120b', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS not in oss
    oss_prefixed = await _run_guidance(tmp_path, system='openrouter', name='openai/gpt-oss-120b', autonomy='max')
    assert _ENVIRONMENT_CONSTRAINTS not in oss_prefixed


async def test_empty_table_disables_default_guidance(tmp_path: Path):
    """An empty table turns the default row off; the function model issues no provider request."""
    text = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='max', model_guidance=())
    assert _MISREAD_CLASSIFY not in text
    assert _ENVIRONMENT_CONSTRAINTS not in text
    assert _MISREAD_MAX in text


async def test_explicit_table_applies_at_high_autonomy(tmp_path: Path):
    """An explicit table applies at every autonomy level; the function model issues no provider request."""
    row = ModelGuidance(
        'openai', replace={'misread': _MISREAD_CLASSIFY}, add=(_STRUCTURED_HANDOFF, _ENVIRONMENT_CONSTRAINTS)
    )
    text = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='high', model_guidance=(row,))
    assert _MISREAD_CLASSIFY in text
    assert _STRUCTURED_HANDOFF in text
    assert _ENVIRONMENT_CONSTRAINTS in text
    assert _MISREAD not in text


async def test_empty_replacement_drops_max_misread_fragment(tmp_path: Path):
    """An empty replacement drops the named fragment; the function model issues no provider request."""
    text = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        autonomy='max',
        model_guidance=(ModelGuidance('openai', replace={'misread': ''}),),
    )
    assert 'assume you misread' not in text
    assert _MISREAD_MAX not in text


async def test_explicit_non_openai_provider_row_matches_system(tmp_path: Path):
    """A non-openai match key compares Model.system exactly; the function model issues no provider request."""
    row = ModelGuidance('anthropic', add=('anthropic-only-guidance',))
    table = (row,)
    anthropic = await _run_guidance(
        tmp_path, system='anthropic', name='claude-opus-4', autonomy='high', model_guidance=table
    )
    assert 'anthropic-only-guidance' in anthropic
    openai = await _run_guidance(tmp_path, system='openai', name='gpt-6.2-sol', autonomy='high', model_guidance=table)
    assert 'anthropic-only-guidance' not in openai


async def test_glob_row_beats_provider_row_and_first_tie_wins(tmp_path: Path):
    """A model glob beats a provider-only row; the first row wins a tie. No provider request."""
    provider_row = ModelGuidance('openai', add=('provider-guidance',))
    model_row = ModelGuidance('openai:gpt-6.2-*', add=('model-guidance',))
    later_glob = ModelGuidance('openai:gpt-6.2-*', add=('later-glob-guidance',))

    glob_second = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        autonomy='high',
        model_guidance=(provider_row, model_row),
    )
    assert 'model-guidance' in glob_second
    assert 'provider-guidance' not in glob_second

    # First glob wins over a later glob and over a later provider-only row.
    glob_first = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        autonomy='high',
        model_guidance=(model_row, later_glob, provider_row),
    )
    assert 'model-guidance' in glob_first
    assert 'later-glob-guidance' not in glob_first
    assert 'provider-guidance' not in glob_first

    # A model that misses the glob still takes the provider-only row.
    other = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-5',
        autonomy='high',
        model_guidance=(provider_row, model_row),
    )
    assert 'provider-guidance' in other
    assert 'model-guidance' not in other

    # A glob also matches a gateway leaf name.
    leaf = await _run_guidance(
        tmp_path,
        system='openrouter',
        name='openai/gpt-6.2-sol',
        autonomy='high',
        model_guidance=(model_row,),
    )
    assert 'model-guidance' in leaf
    miss = await _run_guidance(
        tmp_path,
        system='openrouter',
        name='openai/gpt-5',
        autonomy='high',
        model_guidance=(model_row,),
    )
    assert 'model-guidance' not in miss


async def test_default_table_unwraps_wrapper_and_fallback_models(tmp_path: Path):
    """Wrapper and fallback models resolve to the wrapped / first constituent; no provider request."""
    wrapped = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        wrap=lambda model: WrapperModel(model),
    )
    assert _ENVIRONMENT_CONSTRAINTS in wrapped

    first = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        wrap=lambda model: FallbackModel(model, _NamedFunction('anthropic', 'claude-opus-4', [])),
    )
    assert _ENVIRONMENT_CONSTRAINTS in first

    second = await _run_guidance(
        tmp_path,
        system='anthropic',
        name='claude-opus-4',
        wrap=lambda model: FallbackModel(model, _NamedFunction('openai', 'gpt-6.2-sol', [])),
    )
    assert _ENVIRONMENT_CONSTRAINTS not in second

    wrapped_first = await _run_guidance(
        tmp_path,
        system='openai',
        name='gpt-6.2-sol',
        wrap=lambda model: WrapperModel(FallbackModel(model, _NamedFunction('anthropic', 'claude-opus-4', []))),
    )
    assert _ENVIRONMENT_CONSTRAINTS in wrapped_first

    wrapped_second = await _run_guidance(
        tmp_path,
        system='anthropic',
        name='claude-opus-4',
        wrap=lambda model: WrapperModel(FallbackModel(model, _NamedFunction('openai', 'gpt-6.2-sol', []))),
    )
    assert _ENVIRONMENT_CONSTRAINTS not in wrapped_second


async def test_default_table_adjustments_reach_delegate_guidance(tmp_path: Path):
    """A matching OpenAI max-autonomy run also applies the row to the delegate prompt; no provider request."""
    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.instructions or '')
        first = messages[0].parts[-1]
        prompt = first.content if isinstance(first, UserPromptPart) else None
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart(content='done')])
        if prompt == 'go':
            return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Run it'})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent[object](
        _SystemFunction(respond, system='openai', name='gpt-6.2-sol'),
        capabilities=[LocalWorkspace(tmp_path), Coder[object](autonomy='max', repo_context=False)],
    )
    await agent.run('go')
    delegate = next(text for text in seen if 'You are running a delegated task.' in text)
    assert _MISREAD_CLASSIFY in delegate
    assert _STRUCTURED_HANDOFF in delegate
    assert _ENVIRONMENT_CONSTRAINTS in delegate
    assert _MISREAD_MAX not in delegate


def test_model_guidance_default_table_is_exported():
    """The public default table is a sequence of ModelGuidance rows; no provider request."""
    assert MODEL_GUIDANCE
    assert all(isinstance(row, ModelGuidance) for row in MODEL_GUIDANCE)
    assert MODEL_GUIDANCE[0].match == 'openai'
