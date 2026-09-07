"""The deprecated `fallback_model` spelling of the `fallback_subagent_model` field.

Shared by [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] and
[`XSearch`][pydantic_ai.capabilities.XSearch], the two capabilities that fall back to a subagent
rather than to a local tool of their own.
"""

# TODO(v3): remove this module, along with the `fallback_model` parameter on `ImageGeneration.__init__`
# and `XSearch.__init__` and the deprecated property on each of them.

from __future__ import annotations

import inspect
import warnings

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError


def check_deprecated_fallback_model(
    cls_name: str,
    *,
    # Keyword-only: two booleans swapped by position would make the new name warn and the
    # deprecated one win, with nothing for a type checker to catch.
    fallback_subagent_model_passed: bool,
    fallback_model_passed: bool,
) -> None:
    """Warn on the deprecated `fallback_model` argument and refuse when both spellings are passed.

    Takes only whether each argument was supplied, not the values: the caller keeps the value it
    wants (`fallback_model if fallback_model is not None else fallback_subagent_model`) and this
    only decides whether to warn or raise. Does nothing when `fallback_model` is omitted. When it is
    passed, emits a [`PydanticAIDeprecationWarning`][pydantic_ai.exceptions.PydanticAIDeprecationWarning];
    passing both refuses rather than picking one silently. The exception depends on the entry path:
    direct construction raises [`UserError`][pydantic_ai.exceptions.UserError], while through
    [`Agent.from_spec`][pydantic_ai.agent.Agent.from_spec] the same refusal surfaces as a
    `ValueError` carrying that `UserError` as its `__cause__`, because `_spec.load_from_registry`
    wraps every capability-constructor error.
    """
    if not fallback_model_passed:
        return
    if fallback_subagent_model_passed:
        raise UserError(
            f'{cls_name}: cannot specify both `fallback_model` and `fallback_subagent_model` — '
            '`fallback_model` is the deprecated spelling of `fallback_subagent_model`, so pass only the latter'
        )
    # Attribute the notice to the first frame whose top-level package isn't `pydantic_ai`, so it
    # points at the user whether they constructed the capability directly or loaded an old spec.
    # Compared as a top-level package rather than a prefix: `pydantic_ai_harness` is a caller.
    frame = inspect.currentframe()
    stacklevel = 1
    while frame is not None and frame.f_globals.get('__name__', '').partition('.')[0] == 'pydantic_ai':
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(
        '`fallback_model` is deprecated; use `fallback_subagent_model` instead.',
        PydanticAIDeprecationWarning,
        stacklevel=stacklevel,
    )
