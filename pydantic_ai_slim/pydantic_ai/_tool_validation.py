"""Classification of tool-call argument validation failures ("near-miss" telemetry).

When a model emits a tool call whose arguments fail Pydantic validation, the shape of
the failure is a useful signal: modern models increasingly emit payloads that are
correct except for invented extra keys (see Armin Ronacher, "Better Models: Worse
Tools", https://lucumr.pocoo.org/2026/7/4/better-models-worse-tools/). Today that is an
undifferentiated validation error, so the phenomenon can't be measured.

[`classify_tool_args_validation_failure`][pydantic_ai._tool_validation.classify_tool_args_validation_failure]
is a small pure function that buckets a Pydantic `ValidationError` into a coarse `kind`.
It is deliberately dependency-free and side-effect-free so the eventual argument-repair
layer (which would drop unknown keys, coerce types, etc. before retrying) can reuse the
exact same classification it is instrumented against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

ToolArgsValidationFailureKind = Literal[
    'not_json',
    'unknown_keys_only',
    'type_mismatch_only',
    'missing_required',
    'mixed',
]
"""Coarse classification of why a tool call's arguments failed validation.

- `not_json`: the raw arguments string could not be parsed as JSON at all.
- `unknown_keys_only`: every error is an unexpected/extra key (the Ronacher class).
- `type_mismatch_only`: every error is a type/coercion error (no missing or extra keys).
- `missing_required`: every error is a missing required field.
- `mixed`: any other combination of the above.
"""

# Maximum number of unknown key names to record for an `unknown_keys_only` failure.
# Bounded to keep the telemetry payload small; the fingerprint is the presence and
# identity of the first few invented keys, not an exhaustive list.
_MAX_UNKNOWN_KEYS = 5


@dataclass(frozen=True)
class ToolArgsValidationFailure:
    """The classified shape of a tool-call argument validation failure."""

    kind: ToolArgsValidationFailureKind
    """The coarse failure category."""

    unknown_keys: tuple[str, ...] = field(default=())
    """The names of the unexpected keys, for a `unknown_keys_only` failure (bounded, deduplicated,
    in first-seen order). Empty for every other `kind`."""


def classify_tool_args_validation_failure(error: ValidationError) -> ToolArgsValidationFailure:
    """Classify a Pydantic `ValidationError` raised while validating tool-call arguments.

    Pure and side-effect-free: it only reads `error.errors()`. Shared by the instrumentation
    layer (which records the result on the tool-call span) and, in the future, by an
    argument-repair layer that would act on the same classification.
    """
    errors = error.errors(include_url=False, include_context=False)

    unknown_keys: list[str] = []
    saw_unknown = False
    saw_missing = False
    saw_type_mismatch = False

    for err in errors:
        err_type = err['type']
        if err_type == 'json_invalid':
            # The whole arguments string was unparsable; no field-level errors are meaningful.
            return ToolArgsValidationFailure(kind='not_json')
        elif err_type == 'extra_forbidden':
            saw_unknown = True
            loc = err['loc']
            if loc:  # pragma: no branch - `extra_forbidden` always carries the offending key
                key = str(loc[-1])
                if key not in unknown_keys:
                    unknown_keys.append(key)
        elif err_type == 'missing':
            saw_missing = True
        else:
            # Everything else (int_parsing, string_type, bool_parsing, list_type, ...) is a
            # type/coercion mismatch.
            saw_type_mismatch = True

    if saw_unknown and not saw_missing and not saw_type_mismatch:
        return ToolArgsValidationFailure(
            kind='unknown_keys_only',
            unknown_keys=tuple(unknown_keys[:_MAX_UNKNOWN_KEYS]),
        )
    if saw_missing and not saw_unknown and not saw_type_mismatch:
        return ToolArgsValidationFailure(kind='missing_required')
    if saw_type_mismatch and not saw_unknown and not saw_missing:
        return ToolArgsValidationFailure(kind='type_mismatch_only')
    return ToolArgsValidationFailure(kind='mixed')
