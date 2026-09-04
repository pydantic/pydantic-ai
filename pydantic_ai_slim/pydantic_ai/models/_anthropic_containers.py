"""Anthropic `container_upload` placement and rejected-container retry.

**Placement.** The server only materializes a `container_upload` that falls inside the
turn it is generating — everything after the last completed assistant response. A block
in an earlier finished turn is inert
(https://github.com/pydantic/pydantic-ai/issues/7775). A `tool_use`/`tool_result`
exchange does not end the turn.

The blocks are recomputed from static `CodeExecutionTool.files` every request, so a
message that carries them must carry them on every later request too. Tracking the tail,
or pinning first-and-last, moves the insertion point as history grows and busts the
cacheable prefix.

So they go on every user message except one whose content is only `tool_result` blocks.
That message never opens a turn, so the prompt that did already carries the block. The
skip is also required: `_drop_unpaired_native_tool_calls` keeps an unpaired native call
only while every later message is that shape — the one shape Anthropic accepts when a
client-tool call is mixed with code execution
(https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#how-code-execution-works).
Adding a block there retroactively invalidates that decision and the next request 400s
with `bash_code_execution tool use ... was found without ...`.

Each message gets its own block objects. Later passes mutate blocks in place
(`cache_control`); sharing one dict would smear a write across every message.

**Retry.** Anthropic's documented recovery for a container that cannot be reused:

    Send the request again without the `container` parameter to get a new container.

https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool#container-reuse

The docs do not name the error. The shape we measured is 500 with a generic `api_error`
body (`Internal server error`) when a history-resolved id is paired with `container_upload`
blocks — not the 404 a never-existed id returns. That body is the same envelope as any
other Anthropic 500, so wording cannot distinguish this failure; the guard keys on the
request shape instead. Containers expire 30 days after creation; after ~5 minutes of
inactivity they are checkpointed and a request inside that window restores them.
`expires_at` is a shorter rolling value that does not report the 30-day limit. The recorded
history-resolved id was created on 2026-06-30 and retried on 2026-08-28, about 59 days later,
so it had expired. Its generic 500 with `container_upload` has no typed error or discriminator;
that gap is https://github.com/pydantic/pydantic-ai/issues/7833.

The guard needs both halves of that shape: an id we resolved from history, and uploads on
the wire. A caller-set `anthropic_container` and a `pause_turn` reconnect id stay on the
wire. One retry, not a loop; the client's `max_retries` still multiplies it. If Anthropic
ever returns the typed error the docs imply, widen this guard to it and drop the 500
special case.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .._utils import is_str_dict

if TYPE_CHECKING:
    from anthropic.types.beta import (
        BetaContentBlockParam,
    )


def is_tool_result_only(content: Sequence[BetaContentBlockParam]) -> bool:
    """True when a user message is only `tool_result` blocks — answers a turn, never opens one."""
    return bool(content) and all(is_str_dict(block) and block['type'] == 'tool_result' for block in content)
