"""The stored JSON schema for an `agent__<name>` variable: one half of the contract with Logfire."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# 64 KiB of text is already roughly 16K tokens for typical English prose. It accommodates substantial
# instructions while preventing one managed entry from adding megabytes to every model request.
#
# Counted in Unicode *code points* in every language that implements this contract. Python's `len` and
# JSON Schema's `maxLength` already are; JavaScript's `String.length` is not -- it counts UTF-16 units,
# so an emoji or a CJK extension character weighs two -- which is why the TypeScript core counts
# `[...text].length`. Without that, the same published value fits one core's budget and not the other's.
MAX_MODEL_FACING_TEXT_LENGTH = 65_536

AGENT_CONFIG_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'instructions': {
            'description': (
                'Instruction blocks added to the ones the agent assembles in code, not a replacement for '
                'them. A bare string is one added block. An entry with an `id` swaps out the block the '
                'agent already sends under that key instead of adding one.'
            ),
            'anyOf': [
                {'type': 'string', 'minLength': 1, 'maxLength': MAX_MODEL_FACING_TEXT_LENGTH},
                {
                    'type': 'array',
                    'items': {
                        'anyOf': [
                            {'type': 'string', 'minLength': 1, 'maxLength': MAX_MODEL_FACING_TEXT_LENGTH},
                            {
                                'type': 'object',
                                'properties': {
                                    'id': {
                                        'type': 'string',
                                        'minLength': 1,
                                        'description': (
                                            'The id of the instruction block to address, as the baseline '
                                            'lists it. Omit to add a block instead. A block the baseline '
                                            'marks dynamic cannot be addressed: replacing it would pin one '
                                            'rendering and dropping it would remove the computation, so '
                                            'either is ignored.'
                                        ),
                                    },
                                    'instructions': {
                                        'anyOf': [
                                            {
                                                'type': 'string',
                                                'minLength': 1,
                                                'maxLength': MAX_MODEL_FACING_TEXT_LENGTH,
                                            },
                                            {'type': 'null'},
                                        ],
                                        'description': 'The text to send, or null to drop the addressed block.',
                                    },
                                    'dynamic': {
                                        'type': 'boolean',
                                        'description': (
                                            'Whether the block is recomputed per request. Set on the '
                                            'code-side baseline and ignored on input -- but a block '
                                            'marked true is not addressable, so an editor should offer '
                                            'no override for it.'
                                        ),
                                    },
                                },
                            },
                        ]
                    },
                },
            ],
        },
        'model': {
            'type': 'string',
            'minLength': 1,
            'description': "A model string in 'provider:model' form, such as 'anthropic:claude-fable-5-1'.",
        },
        'settings': {
            'type': 'object',
            'description': (
                'Model settings patch. Only the keys named here are applied; a key an SDK does not know is ignored.'
            ),
            'properties': {
                'max_tokens': {'type': 'integer'},
                'temperature': {'type': 'number'},
                'top_p': {'type': 'number'},
                'top_k': {'type': 'integer'},
                'seed': {'type': 'integer'},
                'presence_penalty': {'type': 'number'},
                'frequency_penalty': {'type': 'number'},
                'parallel_tool_calls': {'type': 'boolean'},
                'timeout': {'type': 'number'},
                'stop_sequences': {'type': 'array', 'items': {'type': 'string'}},
                'thinking': {
                    'anyOf': [{'type': 'boolean'}, {'type': 'string'}],
                    'description': "Enabled/disabled, or an effort level: 'minimal', 'low', 'medium', 'high', 'xhigh'.",
                },
            },
        },
        'tool_definitions': {
            'type': 'array',
            'description': (
                'LLM-facing overlays, each naming the tool it patches by its code-side name. Parameter '
                'names, types, requiredness, validation, and implementation stay code-defined. The baseline '
                'also says which `toolset` each tool came from.'
            ),
            'items': {
                'type': 'object',
                'required': ['name'],
                'properties': {
                    'name': {
                        'type': 'string',
                        'minLength': 1,
                        'description': "The tool's code-side name, which is what this entry patches.",
                    },
                    'new_name': {
                        'type': 'string',
                        'minLength': 1,
                        'description': 'Name shown to the model; a call to it routes back to the original tool.',
                    },
                    'description': {'type': 'string'},
                    'parameters': {
                        'type': 'object',
                        'description': 'Patches per top-level parameter name.',
                        'additionalProperties': {
                            'type': 'object',
                            'properties': {'description': {'type': 'string'}},
                        },
                    },
                    'toolset': {
                        'type': 'string',
                        'description': (
                            'The toolset the tool came from: the baseline reports it, and on an override it '
                            "narrows the match to that toolset's tool of this name."
                        ),
                    },
                },
            },
        },
    },
}
"""The stored JSON schema for an `agent__<name>` variable, shared with the Logfire Agent Control UI.

The Logfire UI holds a copy of this in `app/project/managed-agents/agent-config.ts`, and whichever
side creates the variable first is the one whose schema is persisted. The schema is not cosmetic: the
Logfire backend validates every new version of the value against it, so anything this schema rejects
cannot be written at all.

It is maintained by hand rather than taken from `AgentConfig.model_json_schema()` because the two
artifacts answer different questions. Pydantic's output describes *this* release's model on *this*
Pydantic version -- `$defs`, `anyOf [T, null]` wrappers, `default: null`, and `title` noise -- while
the stored schema is a long-lived contract between a Logfire project and every SDK version that will
ever write to it, in any language.

That is also why it is permissive at every level: `AgentConfig` ignores extra keys precisely so a key
written by a newer UI does not fail validation and revert the whole config to code, and an
`additionalProperties: false` anywhere in the stored schema would defeat that by rejecting the key at
write time instead. For the same reason the fields whose accepted values grow over releases
(`thinking`) are typed rather than enumerated, and `settings` names the canonical keys for the
editor's benefit while leaving unnamed ones writable: a key a newer contract adds has to be storable
before every SDK reading the variable knows it, and `AgentConfigSettings` ignores -- and, under
`OnUnmatched`, reports -- the ones this release cannot apply.

The constraints it does keep are structural rather than versioned, and each one closes a hole a
permissive schema would otherwise leave open. Every `minLength: 1` says the same thing as
`NonEmptyStr`: `''` is a half-filled field, never a value, and `model: ''` in particular takes an
agent down with an unknown-model error on every request. `tool_definitions` items require a `name`
because an overlay that names no tool cannot be applied to anything, so accepting it would only let
the UI save a row that silently does nothing.
"""


def canonical_json(document: dict[str, Any]) -> bytes:
    """The bytes every copy of this contract digests, and the only definition of "canonical" here.

    Sorted keys and `(',', ':')` separators make the document independent of how each copy's literal
    happens to be written. `ensure_ascii=False` makes it independent of the language: `json.dumps`
    escapes non-ASCII by default and `JSON.stringify` does not, so without it the first non-ASCII
    character in a description or a sample value would give two identical documents different digests.

    Public, and not only because `SCHEMA_SHA256` is taken over
    it. Every digest this contract carries has to be taken over the same three flags -- the schema's,
    the one an agent's reported baseline carries as `agent_control.baseline_sha256`, and any an
    adapter computes for itself -- and a second implementation of "canonical" is how two sides come to
    disagree about one document. One function, exported, so there is nothing to restate.
    """
    return json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


SCHEMA_SHA256 = hashlib.sha256(canonical_json(AGENT_CONFIG_JSON_SCHEMA)).hexdigest()
"""SHA-256 of the stored schema's canonical JSON, which every other copy of this contract pins.

The contract has at least three copies -- this package, the TypeScript package, and the Logfire UI's
`agent-config.ts` -- written independently against the same prose, and prose is exactly what lets
them drift: two copies once differed in a description and in whether `id` accepted `null`, a real
difference in what each side would let you save, settled by whichever one happened to create the
variable first. A digest cannot drift quietly, and there is no artifact the repositories share to
hold the schema once. The canonical form is the JSON with sorted keys, `(',', ':')` separators, and
`ensure_ascii=False`, encoded as UTF-8, so every language computes the same digest over the same
bytes. `ensure_ascii=False` is the one of the three that is currently invisible: the schema is
ASCII-only, so it does not move the digest today. It is stated anyway because `json.dumps` escapes
non-ASCII by default and `JSON.stringify` does not, so the first description or sample value with a
non-ASCII character in it would otherwise give the two copies different digests for the same
document -- a drift in the canonical form rather than in the contract, which is the one kind this
digest exists to catch and would be worst at explaining.

When a test pinning this fails, decide which side is right *before* repinning, then change all of
them.
"""
