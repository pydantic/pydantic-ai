"""The model menu: named model options a delegation can be routed to."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings


@dataclass(frozen=True)
class ModelOption:
    """One entry on the model menu, as a model plus how it should run.

    Pass bare model references when the menu keys speak for themselves, and a
    `ModelOption` when an entry needs a routing hint or its own settings:

    ```python
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai_harness.subagents import ModelOption, SubAgents

    SubAgents(
        models={
            'fast': 'anthropic:claude-haiku-4-5',
            'deep': ModelOption(
                'anthropic:claude-opus-4-7',
                description='hard reasoning, multi-file changes',
                settings=ModelSettings(thinking='xhigh'),
            ),
        },
    )
    ```
    """

    model: Model | KnownModelName | str
    """The model a delegation routed to this option runs on."""

    description: str | None = None
    """What this option is for, listed in the prompt next to the key so the parent
    can route on task difficulty rather than on model names alone."""

    settings: ModelSettings | None = None
    """Settings for a delegation routed to this option -- thinking effort,
    temperature, and so on. They merge over the sub-agent's own `model_settings`,
    which keeps whatever the sub-agent set and is not overridden here."""


def as_option(value: Model | KnownModelName | str | ModelOption) -> ModelOption:
    """Normalize one menu value: a bare model reference becomes a plain `ModelOption`."""
    return value if isinstance(value, ModelOption) else ModelOption(value)


def model_label(model: Model | KnownModelName | str) -> str:
    """How a menu entry's model is named in the prompt listing.

    A `Model` instance is labelled `<system>:<model_name>` to match the string form
    users write; a string reference is used as given.
    """
    if isinstance(model, Model):
        return f'{model.system}:{model.model_name}'
    return model


def validate_restriction(agent_name: str, allowed: Sequence[str] | None, menu: Mapping[str, ModelOption]) -> None:
    """Check one delegate's `SubAgent.models` restriction against the configured menu.

    Raises:
        ValueError: the restriction is empty, or names a key the menu does not define.
    """
    if allowed is None:
        return
    if not allowed:
        raise ValueError(
            f'Sub-agent {agent_name!r} has an empty `models` restriction. '
            f'Leave `models` unset to allow every configured model.'
        )
    unknown = [key for key in allowed if key not in menu]
    if unknown:
        available = ', '.join(menu) or '(none configured)'
        raise ValueError(
            f'Sub-agent {agent_name!r} restricts `models` to unknown option(s) {", ".join(repr(k) for k in unknown)}. '
            f'Add them to `SubAgents(models=...)`; configured options: {available}.'
        )
