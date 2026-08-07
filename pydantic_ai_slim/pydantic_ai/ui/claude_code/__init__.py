"""Claude Code CLI `stream-json` output for Pydantic AI agents.

This module maps a Pydantic AI agent's run events onto the JSONL format that
`claude -p --output-format stream-json --verbose` writes to stdout, so that tooling built around the
Claude Code CLI can consume any Pydantic AI agent.

Output only: `stream-json` describes what the CLI wrote, so there is no protocol-specific run input
to accept and therefore no `UIAdapter` counterpart.
"""

from ._event_stream import NDJSON_CONTENT_TYPE, ClaudeCodeEventStream
from ._types import ClaudeCodeEvent

__all__ = [
    'ClaudeCodeEventStream',
    'ClaudeCodeEvent',
    'NDJSON_CONTENT_TYPE',
]
