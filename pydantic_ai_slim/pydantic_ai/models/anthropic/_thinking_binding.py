"""Recovery from a replayed thinking block that Anthropic rejects as bound to a different prefix."""

from __future__ import annotations as _annotations

import warnings
from typing import cast

from anthropic import APIStatusError, Omit, omit as OMIT
from anthropic.types.beta import BetaThinkingBlockBindingParam, BetaThinkingConfigParam

from ... import _utils
from ...messages import ModelMessage, ModelRequest, ModelResponse
from ...profiles import ModelProfile
from ...profiles.anthropic import AnthropicModelProfile
from ._errors import reported_error_body

ANTHROPIC_THINKING_BINDING_BETA = 'thinking-binding-controls-2026-08-01'


_ANTHROPIC_DROP_STALE_THINKING_BLOCKS: BetaThinkingBlockBindingParam = {'prefix_mismatch_behavior': 'drop_block'}


_STALE_THINKING_BLOCK_MARKER = 'The block is bound to a different conversation'


_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'


_ANTHROPIC_COUNT_TOKENS_DROP_STALE_THINKING_BLOCKS_METADATA_KEY = 'anthropic_count_tokens_drop_stale_thinking_blocks'


class AnthropicStaleThinkingBlockWarning(Warning):
    """Warning raised when Anthropic rejected a replayed thinking block and Pydantic AI retried without it.

    Claude Fable 5.1, Claude Opus 5.5, Claude Sonnet 5.5, and Claude Haiku 5.5 bind each thinking block to the
    conversation prefix that produced it and reject a replay once that prefix changes — which a dynamic
    [instructions][pydantic_ai.Agent.instructions] function and a
    [filtered toolset](../toolsets.md#filtering-tools) both do by design. Anthropic enforces the
    check for accounts created on or after 2026-08-31; for older accounts it records the mismatch
    and acts on it only if the request asks it to.

    Pydantic AI therefore asks for nothing by default, so an older account keeps replaying its
    reasoning untouched. Where the check is enforced, the rejected request is retried once with
    `thinking.block_binding.prefix_mismatch_behavior='drop_block'`: the stale block is dropped, the
    run continues, and later requests carrying that response history keep asking for the drop as
    Anthropic requires. The drop is recorded on
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] and as
    an `anthropic.input_transformations` span event.
    """


def can_add_drop_block(thinking: dict[str, object] | Omit) -> bool:
    """Whether a request may add `drop_block` to its wire `thinking` object.

    Not when the caller set a `block_binding` of their own, and not for a thinking type other than
    `adaptive`, since Anthropic accepts `block_binding` only alongside adaptive thinking. A missing
    type counts as adaptive, which is what `drop_stale_thinking_blocks` fills in.
    """
    return isinstance(thinking, Omit) or (
        'block_binding' not in thinking and thinking.get('type', 'adaptive') == 'adaptive'
    )


def is_stale_thinking_block_error(
    profile: ModelProfile,
    thinking: dict[str, object] | Omit,
    error: APIStatusError,
) -> bool:
    """Whether Anthropic rejected a replayed thinking block that the request can be retried without.

    Scoped to models that bind and to requests that set no `block_binding` of their own, through the
    typed `thinking` config or through `extra_body`: an explicit `'error'` is a caller asking to
    fail, and an explicit `'drop_block'` cannot produce this error. A thinking type other than
    `adaptive` is out too; see `can_add_drop_block`.
    """
    if error.status_code != 400 or not profile.get('anthropic_binds_thinking_blocks', False):
        return False
    if not can_add_drop_block(thinking):
        return False
    reported = reported_error_body(error.body)
    return (
        _utils.is_str_dict(reported)
        and isinstance(message := reported.get('message'), str)
        and _STALE_THINKING_BLOCK_MARKER in message
    )


def drop_stale_thinking_blocks(thinking: dict[str, object] | Omit) -> dict[str, object]:
    """The `thinking` object for the retried request, carrying the caller's own config plus the drop.

    A binding model emits thinking blocks whether or not the request configured thinking, so the
    retry usually has no `thinking` object for the binding to ride in, and an `extra_body` one may
    carry no `type`. Claude Sonnet 5.5 and Haiku 5.5 reject a `thinking` object without a `type`, and every binding
    model thinks adaptively when none is given, so `'adaptive'` fills the gap without changing what
    the caller asked for. The retry sends the whole object through `extra_body`, since the SDK's
    discriminated union has no typed home for `block_binding` on every config shape.
    """
    configured: dict[str, object] = {} if isinstance(thinking, Omit) else thinking
    return {'type': 'adaptive', **configured, 'block_binding': _ANTHROPIC_DROP_STALE_THINKING_BLOCKS}


def history_dropped_stale_thinking_blocks(
    messages: list[ModelMessage],
    *,
    compaction_boundary: ModelResponse | None = None,
    include_count_tokens_recovery: bool = False,
) -> bool:
    """Whether Anthropic already told us this conversation needs the drop on later requests."""
    for message in reversed(messages):
        if include_count_tokens_recovery and isinstance(message, ModelRequest) and message.metadata:
            namespace: object = message.metadata.get(_PYDANTIC_AI_METADATA_KEY)
            if (
                _utils.is_str_dict(namespace)
                and namespace.get(_ANTHROPIC_COUNT_TOKENS_DROP_STALE_THINKING_BLOCKS_METADATA_KEY) is True
            ):
                return True
        if message is compaction_boundary or not isinstance(message, ModelResponse) or not message.provider_details:
            continue
        transformations: object = message.provider_details.get('input_transformations')
        if isinstance(transformations, list) and any(
            _utils.is_str_dict(transformation)
            and transformation.get('type') == 'thinking_dropped'
            and transformation.get('reason') == 'prefix_binding_mismatch'
            for transformation in cast(list[object], transformations)
        ):
            return True
    return False


def mark_history_to_drop_stale_thinking_blocks_for_count_tokens(messages: list[ModelMessage]) -> None:
    """Keep a successful count-token recovery active for later counts of this conversation."""
    request = next((message for message in reversed(messages) if isinstance(message, ModelRequest)), None)
    if request is None:  # pragma: no cover
        return
    request.metadata = request.metadata or {}
    namespace: object = request.metadata.get(_PYDANTIC_AI_METADATA_KEY)
    if not _utils.is_str_dict(namespace):
        namespace = {}
        request.metadata[_PYDANTIC_AI_METADATA_KEY] = namespace
    namespace[_ANTHROPIC_COUNT_TOKENS_DROP_STALE_THINKING_BLOCKS_METADATA_KEY] = True


def thinking_with_stale_block_history(
    profile: AnthropicModelProfile,
    messages: list[ModelMessage],
    thinking: BetaThinkingConfigParam | Omit,
    effective_thinking: dict[str, object] | Omit,
    betas: set[str],
    *,
    compaction_boundary: ModelResponse | None = None,
    include_count_tokens_recovery: bool = False,
) -> tuple[BetaThinkingConfigParam | Omit, set[str], dict[str, object] | None, dict[str, object] | Omit]:
    """Resolve request parameters that keep a prior request-local drop active for this history."""
    keep_dropping = (
        profile.get('anthropic_binds_thinking_blocks', False)
        and can_add_drop_block(effective_thinking)
        and history_dropped_stale_thinking_blocks(
            messages,
            compaction_boundary=compaction_boundary,
            include_count_tokens_recovery=include_count_tokens_recovery,
        )
    )
    if not keep_dropping:
        return thinking, betas, None, effective_thinking
    thinking_override = drop_stale_thinking_blocks(effective_thinking)
    return OMIT, betas | {ANTHROPIC_THINKING_BINDING_BETA}, thinking_override, thinking_override


def warn_stale_thinking_block_recovery(model_name: str) -> None:
    warnings.warn(
        f'{model_name!r} rejected a replayed thinking block: the conversation prefix changed '
        'between requests, which Anthropic enforces for accounts created on or after 2026-08-31. '
        "Pydantic AI retried with `prefix_mismatch_behavior='drop_block'`, so the run continued "
        "without that turn's reasoning. The cause is something earlier in the request that is not "
        'byte-stable between turns — commonly instructions carrying a per-request value such as a '
        'timestamp, or a tool added, removed or reordered mid-conversation. That same instability '
        'invalidates the prompt cache on every model and provider, where it raises nothing and '
        'silently re-charges the whole conversation at uncached rates, so this rejection is the '
        'loud version of a normally-silent cost. Making the prefix stable is the real fix; to skip '
        'the rejected request instead, ask for the drop yourself: `model_settings='
        "{'anthropic_thinking': {'type': 'adaptive', 'block_binding': "
        "{'prefix_mismatch_behavior': 'drop_block'}}}`. To keep the retry and silence this warning: "
        "`warnings.simplefilter('ignore', AnthropicStaleThinkingBlockWarning)`. See "
        'https://pydantic.dev/docs/ai/models/anthropic/#thinking-block-binding and '
        'https://platform.claude.com/docs/en/build-with-claude/preserved-thinking',
        AnthropicStaleThinkingBlockWarning,
        stacklevel=2,
    )
