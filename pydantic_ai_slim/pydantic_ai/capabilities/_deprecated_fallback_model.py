"""The deprecated `fallback_model` spelling of the `fallback_subagent_model` field.

Shared by [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] and
[`XSearch`][pydantic_ai.capabilities.XSearch], the two capabilities that fall back to a subagent
rather than to a local tool of their own.
"""

# TODO(v3): remove this module, along with the `fallback_model` parameter on `ImageGeneration.__init__`
# and `XSearch.__init__` and the deprecated property on each of them.

from __future__ import annotations

import warnings
from typing import TypeVar

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError

FallbackSubagentModelT = TypeVar('FallbackSubagentModelT')


def resolve_fallback_subagent_model(
    cls_name: str,
    fallback_subagent_model: FallbackSubagentModelT,
    fallback_model: FallbackSubagentModelT,
) -> FallbackSubagentModelT:
    """Map the deprecated `fallback_model` argument onto `fallback_subagent_model`.

    Returns `fallback_subagent_model` unchanged when `fallback_model` is omitted (`None`). When
    `fallback_model` is passed, emits a
    [`PydanticAIDeprecationWarning`][pydantic_ai.exceptions.PydanticAIDeprecationWarning] and returns
    its value; passing both raises [`UserError`][pydantic_ai.exceptions.UserError] rather than
    picking one silently.
    """
    if fallback_model is None:
        return fallback_subagent_model
    if fallback_subagent_model is not None:
        raise UserError(
            f'{cls_name}: cannot specify both `fallback_model` and `fallback_subagent_model` — '
            '`fallback_model` is the deprecated spelling of `fallback_subagent_model`, so pass only the latter'
        )
    # user → `__init__` → here → `warn`; `from_spec` adds a frame and so lands one short, as the
    # other notices these capabilities emit from `__post_init__` already do.
    warnings.warn(
        '`fallback_model` is deprecated; use `fallback_subagent_model` instead.',
        PydanticAIDeprecationWarning,
        stacklevel=3,
    )
    return fallback_model
