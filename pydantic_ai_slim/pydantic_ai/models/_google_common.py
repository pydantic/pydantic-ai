"""Utilities shared by the Google API adapters."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..exceptions import UserError
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..profiles import ModelProfile, merge_profile
from ..profiles.google import (
    GOOGLE_THINKING_LEVEL_SCALE,
    GOOGLE_THINKING_LEVELS,
    GoogleModelProfile,
    GoogleThinkingLevel,
)
from ..providers._model_deprecations import GEMINI_API_MODEL_DEPRECATIONS, GOOGLE_CLOUD_MODEL_DEPRECATIONS
from ..settings import ThinkingEffort

if TYPE_CHECKING:
    from . import Model, ModelRequestParameters


def character_index(text: str, byte_index: int) -> int | None:
    """Convert a Gemini UTF-8 byte offset to a Python string index."""
    encoded = text.encode()
    if not 0 <= byte_index <= len(encoded):
        return None
    try:
        return len(encoded[:byte_index].decode())
    except UnicodeDecodeError:
        return None


def thinking_effort_to_level(thinking: ThinkingEffort) -> GoogleThinkingLevel:
    """Normalize unified thinking effort to a Gemini thinking level."""
    level_by_effort: dict[ThinkingEffort, GoogleThinkingLevel] = {
        'minimal': 'MINIMAL',
        'low': 'LOW',
        'medium': 'MEDIUM',
        'high': 'HIGH',
        # Models whose level set lacks `XHIGH` snap these to `HIGH`. `google-genai`'s `ThinkingLevel` doesn't list
        # `XHIGH` yet, so the SDK warns `XHIGH is not a valid ThinkingLevel` and sends it anyway.
        'xhigh': 'XHIGH',
        'max': 'XHIGH',
    }
    return level_by_effort[thinking]


_GOOGLE_THINKING_LEVEL_ORDER: dict[GoogleThinkingLevel, int] = {
    level: order for order, level in enumerate(GOOGLE_THINKING_LEVEL_SCALE)
}


def snap_thinking_level(
    level: GoogleThinkingLevel, levels: frozenset[GoogleThinkingLevel] | None
) -> GoogleThinkingLevel:
    """Snap a thinking level to the nearest one the model accepts, on the `MINIMAL < LOW < MEDIUM < HIGH < XHIGH` scale.

    Equidistant levels round down to the cheaper one. `None` means the model takes the default
    `GOOGLE_THINKING_LEVELS`. Shared with the Live path, which differs only in where its level set
    comes from.
    """
    if levels is None:
        levels = GOOGLE_THINKING_LEVELS
    if not levels:
        raise UserError('`google_thinking_levels` must contain at least one level when `thinking` is set')
    if unknown := levels - frozenset(GOOGLE_THINKING_LEVEL_SCALE):
        raise UserError(
            f'`google_thinking_levels` contains unknown levels: {sorted(unknown)!r}; '
            f'expected a subset of {sorted(GOOGLE_THINKING_LEVEL_SCALE)!r}'
        )
    requested = _GOOGLE_THINKING_LEVEL_ORDER[level]
    return min(
        levels,
        key=lambda candidate: (
            abs(_GOOGLE_THINKING_LEVEL_ORDER[candidate] - requested),
            _GOOGLE_THINKING_LEVEL_ORDER[candidate],
        ),
    )


def resolve_google_thinking_level(thinking: ThinkingEffort, profile: GoogleModelProfile) -> GoogleThinkingLevel:
    """Map unified thinking to the closest thinking level the model supports."""
    levels = profile.get('google_thinking_levels')
    if levels is None:
        # Sparse profile without a level set: fall back to the boolean floor flag.
        levels = (
            GOOGLE_THINKING_LEVELS
            if profile.get('google_supports_minimal_thinking_level', True)
            else GOOGLE_THINKING_LEVELS - {'MINIMAL'}
        )
    return snap_thinking_level(thinking_effort_to_level(thinking), levels)


GEMINI_API_THINKING_LEVELS: tuple[tuple[str, frozenset[GoogleThinkingLevel]], ...] = (
    # Thinking levels that only the Gemini API enforces, so they follow the client's transport, which
    # `google_model_profile` can't see from the model name; Google Cloud keeps the default levels.
    # Google documents only `minimal` and `high` on both APIs, but the Gemini API alone 400s `LOW` and `MEDIUM`
    # for this id and its `-preview`: verified live 2026-09-30 on GenerateContent (Vertex `global`, `us` and `eu`
    # accept them) and 2026-11-19 on `/interactions` with `store=false`.
    # https://ai.google.dev/gemini-api/docs/image-generation
    ('gemini-3.1-flash-image', frozenset(('MINIMAL', 'HIGH'))),
    # The Gemini API 400s `Thinking level MINIMAL is not supported for this model` and accepts the rest: verified
    # live 2026-11-18 on GenerateContent and 2026-11-19 on `/interactions`. Vertex is not probed and its thinking
    # table doesn't list the model.
    # https://ai.google.dev/gemini-api/docs/thinking#thinking-levels
    # https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking
    ('gemini-4-neon', frozenset(('LOW', 'MEDIUM', 'HIGH', 'XHIGH'))),
)


def with_gemini_api_thinking_levels(model_name: str, profile: GoogleModelProfile) -> GoogleModelProfile:
    """`profile` with the thinking levels the Gemini API enforces for `model_name`, unless `profile` sets its own.

    Apply it only when the client talks to the Gemini API.
    """
    levels = next((levels for prefix, levels in GEMINI_API_THINKING_LEVELS if model_name.startswith(prefix)), None)
    if levels is None or 'google_thinking_levels' in profile:
        return profile
    return {**profile, 'google_thinking_levels': levels, 'google_supports_minimal_thinking_level': 'MINIMAL' in levels}


def with_deprecation(model_name: str, profile: ModelProfile, *, google_cloud: bool) -> ModelProfile:
    """`profile` with the deprecation that Google Cloud's or the Gemini API's schedule gives `model_name`, if any.

    The two schedules date the same model differently. `google-genai` sends a resource name such as `models/<id>` or
    `publishers/google/models/<id>` as is, where it would otherwise add that prefix to a bare id, so both spellings
    reach the same model.
    """
    deprecations = GOOGLE_CLOUD_MODEL_DEPRECATIONS if google_cloud else GEMINI_API_MODEL_DEPRECATIONS
    deprecation = deprecations.get(model_name.rsplit('/', 1)[-1])
    return profile if deprecation is None else merge_profile(profile, ModelProfile(deprecation=deprecation))


def set_server_timeout(headers: dict[str, str], client_timeout_ms: int | None) -> None:
    """Ask Google for a whole-request deadline with `X-Server-Timeout`, unless `headers` already carries one.

    A read timeout only bounds the wait between bytes, so the server would cut off a response that streams for
    longer than its own deadline. Ask for the client's timeout, or Google's long-decode figure if that's longer; a
    client without a timeout sends none. A caller's header keeps its value, in the spelling the GenerateContent SDK
    checks before sending its own.
    """
    # Imported here: `providers.google` needs `google-genai`, which importing this module must not.
    from ..providers.google import _LONG_DECODE_TIMEOUT  # pyright: ignore[reportPrivateUsage]

    if caller_name := next((name for name in headers if name.lower() == 'x-server-timeout'), None):
        headers['X-Server-Timeout'] = headers.pop(caller_name)
    elif client_timeout_ms:
        headers['X-Server-Timeout'] = str(math.ceil(max(client_timeout_ms / 1000, _LONG_DECODE_TIMEOUT)))


def rejects_native_tool(model: Model, params: ModelRequestParameters, tool_type: type[AbstractNativeTool]) -> bool:
    """Whether `model`'s base validation will reject a `tool_type` native tool in `params` as unsupported.

    It does for one the profile's `supported_native_tools` leaves out, that isn't `optional` and has no local fallback,
    as `Model.prepare_request` decides. The Google models check this first, to name the other Google model instead.
    """
    if issubclass(tool_type, tuple(model.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS))):
        return False
    fallback_ids = {t.unless_native for t in params.function_tools if t.unless_native}
    return any(
        isinstance(t, tool_type) and not t.optional and t.unique_id not in fallback_ids for t in params.native_tools
    )
