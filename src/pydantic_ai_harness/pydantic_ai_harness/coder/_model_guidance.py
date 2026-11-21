"""Per-provider and per-model adjustments to `Coder`'s default guidance.

Some models do worse with the shared default text than they could. Rather than tune the shared
text for one model at another's expense, `Coder` looks the run's model up in a small table and,
when a row matches, swaps named fragments of the default text or appends sentences to it.

The default table applies only at `'max'` autonomy (headless), and only to the OpenAI family.
At every other autonomy level, and for every other model, the prompt is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Literal

from pydantic_ai.models import AbstractModel
from pydantic_ai.models.fallback import FallbackModel

Fragment = Literal['misread']
"""Default fragments a row may replace. An empty replacement drops the fragment."""

_OPENAI_MODEL_GLOBS = ('gpt-*', 'chatgpt-*', 'o[1-9]*')
"""Leaf model-name patterns that count as OpenAI family when the provider is not `openai`."""


@dataclass(frozen=True)
class ModelGuidance:
    """One row of the lookup table.

    `match` is a provider name, as in `Model.system` (`'anthropic'`), optionally followed by `:` and a
    glob over `Model.model_name` (`'openai:gpt-6.2-*'`). A row with a model glob beats a provider-only
    row. If several rows have the same specificity, the first one wins.

    The `'openai'` match key means the OpenAI family, not only the `openai` provider: the `openai`
    provider itself, a model name with an `openai/` prefix, or a leaf name matching `gpt-*`,
    `chatgpt-*`, or `o[1-9]*`, except `gpt-oss-*`. Every other provider key compares exactly against
    `Model.system`. A model glob is tried against the full name and the last `/` segment.
    """

    match: str
    replace: Mapping[Fragment, str] = field(default_factory=dict[Fragment, str])
    add: Sequence[str] = ()


MISREAD_CLASSIFY = (
    'If inputs look inconsistent, first check whether the task or its rules classify them '
    '(for example as malformed, conflicting, or rejected) and apply that rule. '
    'Only when no rule covers them, assume you misread: re-read, and try other interpretations before settling.'
)
"""For models that otherwise reinterpret deliberately malformed inputs instead of flagging them."""

STRUCTURED_HANDOFF = (
    'When asked for notes or a handoff, use headed sections: each decision and why, '
    'the exact verification commands with their observed results, the remaining risk, '
    'and what was out of scope or not done.'
)
"""For models whose notes leave out the evidence a reviewer needs."""

ENVIRONMENT_CONSTRAINTS = (
    'When a task states constraints on where or how the code will run '
    '(a restricted or sandboxed interpreter, no dependencies, a specific Python version, pure functions, no I/O), '
    "find out what's actually allowed before writing code. Then prefer the simplest constructs that fit, "
    'and if possible check the code under those constraints.'
)
"""For models that otherwise ignore stated runtime constraints when choosing language features."""

MODEL_GUIDANCE: tuple[ModelGuidance, ...] = (
    ModelGuidance('openai', replace={'misread': MISREAD_CLASSIFY}, add=(STRUCTURED_HANDOFF, ENVIRONMENT_CONSTRAINTS)),
)
"""The table `Coder` uses by default, applied only at `'max'` autonomy."""


def is_openai_family(model: AbstractModel) -> bool:
    """True for OpenAI models, including the same models reached through a gateway.

    Matches the `openai` provider, an `openai/` model-name prefix, and the leaf-name patterns in
    `_OPENAI_MODEL_GLOBS`, except `gpt-oss-*` (open-weights models that share a `gpt-` prefix).
    An OpenAI-compatible server that uses `OpenAIProvider` also reports `system == 'openai'`, so it
    matches. An OpenRouter Anthropic route does not.
    """
    model = _unwrap_model(model)
    leaf = model.model_name.rsplit('/', 1)[-1]
    if fnmatchcase(leaf, 'gpt-oss-*'):
        return False
    if model.system == 'openai' or model.model_name.startswith('openai/'):
        return True
    return any(fnmatchcase(leaf, pat) for pat in _OPENAI_MODEL_GLOBS)


def _unwrap_model(model: AbstractModel) -> AbstractModel:
    if isinstance(model, FallbackModel):
        return _unwrap_model(model.models[0])
    return model


def _provider_matches(provider: str, model: AbstractModel) -> bool:
    if provider == 'openai':
        return is_openai_family(model)
    return provider == model.system


def _model_name_matches(name: str, pattern: str) -> bool:
    if fnmatchcase(name, pattern):
        return True
    leaf = name.rsplit('/', 1)[-1]
    return leaf != name and fnmatchcase(leaf, pattern)


def resolve_model_guidance(table: Sequence[ModelGuidance], model: AbstractModel) -> ModelGuidance | None:
    """The most specific row matching `model`, or `None`.

    A row with a model glob beats a provider-only row. If several rows have the same specificity,
    the first one wins. Wrapper models already report the wrapped model's `system` and `model_name`.
    Fallback models are resolved against their first constituent.
    """
    model = _unwrap_model(model)
    best: ModelGuidance | None = None
    for row in table:
        provider, _, pattern = row.match.partition(':')
        if not _provider_matches(provider, model):
            continue
        if pattern and not _model_name_matches(model.model_name, pattern):
            continue
        if best is None or (pattern and ':' not in best.match):
            best = row
    return best
