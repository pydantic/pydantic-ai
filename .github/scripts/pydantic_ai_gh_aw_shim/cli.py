r"""Pydantic AI gh-aw shim — Claude Code CLI compatibility for gh-aw.

gh-aw runs the agent engine like the Claude Code CLI:

    <command> --print --no-chrome --allowed-tools '<csv>' --debug-file <path> \\
      --verbose --permission-mode <mode> --output-format stream-json \\
      --mcp-config <mcp-servers.json> --prompt-file <prompt.txt> \\
      [--model <model>] "<rendered prompt>"

With `engine.command` set, `<command>` is this shim. It speaks Claude
Code's argv, recovers the prompt, builds a `pydantic-ai` agent backed by
the gh-aw-injected Anthropic-compatible proxy, exposes Claude-named
tools plus gh-aw's MCP servers (GitHub + the `safeoutputs` write-sink),
enforces gh-aw's `--allowed-tools` allow-list, and emits Claude-compatible
`stream-json` so gh-aw's log parser and token accounting keep working.

Like Claude Code itself, the shim only talks to Anthropic-shape APIs
(`ANTHROPIC_BASE_URL` → real Anthropic, MiniMax's Anthropic-compatible
endpoint, etc.). No OpenAI path — the workflow's `engine.id: claude`
contract is Anthropic-shape end to end.

Credentials note: under gh-aw the real API key is *excluded* from the
agent container (`awf --exclude-env ANTHROPIC_API_KEY`). The AWF
api-proxy injects it transparently; the shim only ever sends a
placeholder bearer to the proxy base URL — never a real upstream key.

This module is loaded as the `pydantic_ai_gh_aw_shim.cli` submodule;
`__main__.py` is a 3-line entry stub that calls `cli.main()`. Tests
import this module directly (`from pydantic_ai_gh_aw_shim import cli`),
which is why the runner stub doesn't live in `__main__.py` — running it
under `runpy.run_module(..., run_name="__main__")` plus PEP-563
annotations breaks pydantic-ai's `takes_run_context` detection.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import pathlib
import sys
import time
import uuid
import warnings
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import httpx
import logfire
from anthropic import AsyncAnthropic
from pydantic import ValidationError
from pydantic_ai_harness.experimental import HarnessExperimentalWarning

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeTool, ProcessEventStream
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultEvent,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.native_tools import WebFetchTool
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage, UsageLimits

with warnings.catch_warnings():
    warnings.simplefilter('ignore', HarnessExperimentalWarning)
    from pydantic_ai_harness.experimental.compaction import (
        ClearToolResults,
        DeduplicateFileReads,
        SummarizingCompaction,
        TieredCompaction,
    )

from . import (
    CLAUDE_CODE_TOOL_NAMES,
    MUTATING_TOOLS,
    READ_ONLY_SUBAGENT_TOOLS,
    build_claude_code_toolset,
)
from .shared import logger, reset_context_state

# Type aliases for the public surface — the shim runs `None`-deps agents
# throughout, so every `RunContext` is concretely `RunContext[object]`.
ToolPredicate: TypeAlias = Callable[[RunContext[object], ToolDefinition], bool | Awaitable[bool]]
TaskCallable: TypeAlias = Callable[[RunContext[object], str, str], Awaitable[str]]

# Placeholder bearer token sent to the AWF api-proxy. The proxy strips this
# header and injects the real `ANTHROPIC_API_KEY` on the outbound wire — so
# the agent container never sees the real key. Sent verbatim only when no
# `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` env is provided locally.
PROXY_BEARER_PLACEHOLDER = 'gh-aw-proxy-injected'


def _anthropic_native_capabilities() -> list[NativeTool]:
    """`NativeTool(WebFetchTool())` for real Anthropic only.

    Anthropic-compatible endpoints (MiniMax, etc.) reject the
    `web_fetch_20250910` server-side tool with `invalid_request_error
    (2013)` because they don't implement Anthropic's server-side tool
    types. Detect via `ANTHROPIC_BASE_URL` — empty/unset means the
    Anthropic SDK default (real Anthropic).
    """
    base_url = os.environ.get('ANTHROPIC_BASE_URL', '')
    if not base_url or 'api.anthropic.com' in base_url:
        return [NativeTool(WebFetchTool())]
    return []


# pydantic-ai's built-in request_limit default of 50 is too low for the
# deep multi-step workflows here; gh-aw's api-proxy still caps the run.
REQUEST_LIMIT = 200
SUBAGENT_REQUEST_LIMIT = 75

# Per-request HTTP timeout for every LLM call.  The read timeout is the
# critical one: MiniMax's proxy can hold a streaming connection open without
# sending data.  5 min is generous enough for large generations but prevents
# indefinite hangs.  SDK-level retries cover transient 429/5xx before raising.
_LLM_TIMEOUT = httpx.Timeout(timeout=120.0, connect=10.0)
_LLM_MAX_RETRIES = 4

# Wall-clock caps (seconds).  These are last-resort guards on top of the
# per-request timeout so a burst of slow requests can't accumulate forever.
RUN_TIMEOUT_SECS = 28 * 60  # 28 min — just under the 30 min gh-aw job cap
SUBAGENT_TIMEOUT_SECS = 15 * 60  # 15 min per Task sub-agent

# Static prefix for `Agent(instructions=[INSTRUCTIONS, prompt])`. Sequence
# form lets Anthropic's prompt-prefix cache hit `INSTRUCTIONS` across runs.
INSTRUCTIONS = (
    '## Parallel tool calls\n\n'
    'The model supports parallel tool calls. When multiple reads, searches, or '
    "lookups are independent — meaning one doesn't need another's result — "
    'issue them all in the same response. They execute concurrently. Only '
    'chain sequentially when one call genuinely needs a previous result.\n\n'
    '## File reading\n\n'
    'Read files in large ranges (500+ lines per call). MAX_TOOL_OUTPUT is '
    '50 000 chars so most Python source files fit in one or two reads. '
    'Avoid reading 30–80 lines at a time.\n\n'
    '## Search tools\n\n'
    'Use the native Grep and Glob tools for codebase search. '
    '`rg` and `uv` are also available as plain commands via Bash.\n\n'
    '## Dev environment\n\n'
    'The repo is checked out at $GITHUB_WORKSPACE. Dev dependencies are NOT '
    'pre-installed — run `make install` once before using pytest, ruff, or '
    'pyright. Prefer `uv run pytest <test_file>` over a bare `pytest` call; '
    'uv handles the virtual env automatically.\n\n'
    '## GitHub issue search\n\n'
    '`gh issue list --search` returns HTTP 403 via the AWF firewall proxy. '
    'Use the MCP tools instead: '
    '`mcp__github__search_issues(query="repo:pydantic/pydantic-ai <keywords>")`.'
)

# The real task spec rides in `instructions=`; the user message is a trigger.
RUN_TRIGGER = 'Begin the task per the instructions above.'

SUBAGENT_INSTRUCTIONS = (
    'You are a focused, read-only sub-agent. You can read files, search the '
    'codebase, and fetch web content, but you cannot modify the workspace or '
    'shell out. Investigate the task you were given and return a concise, '
    'evidence-grounded answer to your caller — do not try to act on it.'
)


# History compaction. `TieredCompaction` keeps the conversation within the model's
# window by escalating cheap-to-expensive: dedupe superseded file reads, clear old
# tool results, then -- only if still over `COMPACTION_TARGET_TOKENS` -- summarise
# the older messages behind the recent tail. `Agent(instructions=...)` is re-applied
# on every request, so the workflow prompt never enters the message list and is never
# compacted.

# ~100k tokens (half of a 200k window) at ~4 chars/token.
COMPACTION_TARGET_TOKENS = 100_000
# Messages kept verbatim as the recent tail; older ones feed the summariser.
COMPACTION_KEEP_RECENT = 10

COMPACTION_SUMMARY_INSTRUCTIONS = (
    'Summarise the agent transcript below for resumption in a fresh '
    'context window. Produce a structured brief, not free prose. Use this '
    'exact section layout, omitting any section that is empty:\n\n'
    '## Goal\n'
    'One short paragraph: what the agent was asked to do.\n\n'
    '## Files inspected\n'
    '- `<full/path>`: one-line note on what was found there.\n\n'
    '## Commands run\n'
    '- `<command>`: outcome in one line.\n\n'
    '## Errors encountered\n'
    'Verbatim error messages or unexpected behaviour, with the file or '
    'command that triggered each.\n\n'
    '## Decisions and approaches\n'
    '- Concrete decisions with reasoning. Include approaches already tried '
    'that did **not** work, so they are not re-attempted.\n\n'
    '## Open questions\n'
    '- Anything still unresolved.\n\n'
    '## Next step\n'
    'The single most likely next action.\n\n'
    'Preserve specifics (paths, identifiers, exact strings) over prose. '
    'Respond with text only — do not call any tools.\n\n'
    '<messages>\n{messages}\n</messages>'
)


def _read_file_key(call: ToolCallPart) -> str | None:
    """Identity of a `Read` call for `DeduplicateFileReads`: path plus the read slice.

    A later `Read` of the same `(file_path, offset, limit)` supersedes the earlier
    one, so the earlier result is blanked. Non-`Read` calls return `None` (skipped).
    """
    if call.tool_name != 'Read':
        return None
    args = call.args_as_dict()
    if not isinstance(args, dict):
        return None
    path = args.get('file_path')
    if not isinstance(path, str):
        return None
    return f'{path}|{args.get("offset")!r}|{args.get("limit")!r}'


def _history_compaction() -> TieredCompaction[object]:
    """The run's history-compaction capability.

    Replaces the shim's former hand-rolled two-stage compaction with the harness's
    escalation default: the cheap zero-LLM tiers (dedupe superseded reads, clear old
    tool results) run first, and an LLM summary is paid for only when the history is
    still over `COMPACTION_TARGET_TOKENS`. The summariser inherits the run's model
    and keeps the last `COMPACTION_KEEP_RECENT` messages verbatim.
    """
    # Each tier's own `max_*` trigger is bypassed inside `TieredCompaction` (the
    # orchestrator drives `compact` directly), so it only needs to be valid.
    return TieredCompaction[object](
        tiers=[
            DeduplicateFileReads[object](file_key=_read_file_key),
            ClearToolResults[object](max_tokens=1, keep_pairs=COMPACTION_KEEP_RECENT // 2),
            SummarizingCompaction[object](
                max_messages=1,
                keep_messages=COMPACTION_KEEP_RECENT,
                summary_prompt=COMPACTION_SUMMARY_INSTRUCTIONS,
            ),
        ],
        target_tokens=COMPACTION_TARGET_TOKENS,
    )


@dataclass(slots=True)
class Args:
    """The subset of Claude Code's CLI surface the shim acts on."""

    model: str | None = None
    mcp_config: str | None = None
    prompt_file: str | None = None
    prompt_positional: str | None = None
    # None = flag absent (local/dev: no restriction). A set = enforce it.
    allowed_tools: frozenset[str] | None = None
    permission_mode: str | None = None


def _split_allowed_tools(value: str | None) -> frozenset[str] | None:
    """Parse Claude's `--allowed-tools` CSV into base tool names.

    Entries may carry a permission scope, e.g. `Edit(/tmp/*)` or
    `Bash(git:*)` — only the base name gates availability here, so the
    parenthesised scope is stripped. Returns `None` when the flag is absent
    so non-gh-aw/local runs keep every tool.
    """
    if value is None:
        return None
    names: set[str] = set()
    for raw in value.split(','):
        entry = raw.strip()
        if not entry:
            continue
        names.add(entry.split('(', 1)[0].strip())
    return frozenset(names)


def parse_args(argv: Sequence[str]) -> Args:
    """Parse Claude Code's CLI surface into `Args`, tolerating unknown flags so a future Claude flag never breaks the engine."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--model')
    p.add_argument('--mcp-config')
    p.add_argument('--prompt-file')
    p.add_argument('--output-format', default='stream-json')
    p.add_argument('--allowed-tools')
    p.add_argument('--permission-mode')
    p.add_argument('--debug-file')
    for flag in ('--print', '--no-chrome', '--verbose', '--continue'):
        p.add_argument(flag, action='store_true')
    known, unknown = p.parse_known_args(list(argv))
    # gh-aw appends the rendered prompt as the trailing positional argument.
    positionals = [a for a in unknown if not a.startswith('-')]
    return Args(
        model=known.model,
        mcp_config=known.mcp_config,
        prompt_file=known.prompt_file,
        prompt_positional=positionals[-1] if positionals else None,
        allowed_tools=_split_allowed_tools(known.allowed_tools),
        permission_mode=known.permission_mode,
    )


def resolve_prompt(args: Args) -> str:
    """Prompt precedence: trailing positional -> --prompt-file -> $GH_AW_PROMPT."""
    if args.prompt_positional:
        return args.prompt_positional
    path = args.prompt_file or os.environ.get('GH_AW_PROMPT')
    if path and os.path.isfile(path):
        return pathlib.Path(path).read_text(encoding='utf-8')
    return ''


def build_model(args: Args) -> tuple[Model, str]:
    """Build the `pydantic-ai` model and a human-readable label.

    Anthropic-only — the shim behaves like the stock Claude Code CLI:
    gh-aw sets `ANTHROPIC_BASE_URL` (its in-cluster transparent proxy)
    and the AWF api-proxy injects the real key on outgoing requests.

    **Why we construct `AsyncAnthropic` ourselves** instead of letting
    `pydantic-ai`'s `AnthropicProvider` auto-configure: gh-aw runs the
    agent step in a sandbox that excludes `ANTHROPIC_API_KEY` from the
    container env (`awf --exclude-env ANTHROPIC_API_KEY` — a security
    measure so the real key never reaches the agent). `pydantic-ai`'s
    auto-config requires that env var to be present, so it errors out
    under gh-aw. The explicit `AsyncAnthropic(auth_token=...)` path
    sends a placeholder bearer that the AWF api-proxy swaps for the
    real key on the wire — the same dance the Claude Code CLI does.
    This is a gh-aw constraint, not a pydantic-ai one; upstream gh-aw
    could lift it by allowing the agent to read the key directly, but
    that would break the credential-isolation guarantee.

    Model name resolution (in priority order):
      1. `--model X` argv flag (from Claude Code's CLI surface).
      2. `ANTHROPIC_MODEL` env var (standard Anthropic SDK convention;
         gh-aw populates this from the workflow's `engine.model:` field).
      3. Fallback default `claude-sonnet-4-6`.
    """
    model_name = args.model or os.environ.get('ANTHROPIC_MODEL') or 'claude-sonnet-4-6'
    anthropic_base = os.environ.get('ANTHROPIC_BASE_URL')
    auth_token = (
        os.environ.get('ANTHROPIC_AUTH_TOKEN') or os.environ.get('ANTHROPIC_API_KEY') or PROXY_BEARER_PLACEHOLDER
    )
    logger.info('anthropic model=%s base_url=%s', model_name, anthropic_base or '(default)')
    client = AsyncAnthropic(
        auth_token=auth_token,
        base_url=anthropic_base,
        timeout=_LLM_TIMEOUT,
        max_retries=_LLM_MAX_RETRIES,
    )
    return (
        AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client)),
        f'anthropic:{model_name}',
    )


def configure_logging() -> None:
    """Configure stderr logging once, at CLI entry."""
    logging.basicConfig(
        level=logging.INFO,
        format='[pydantic-ai-gh-aw-shim] %(message)s',
        stream=sys.stderr,
    )


def configure_observability() -> None:
    """Wire pydantic-ai + httpx + mcp instrumentation to Logfire/OTLP if configured."""
    write_token = os.environ.get('LOGFIRE_WRITE_TOKEN') or os.environ.get('LOGFIRE_TOKEN')
    if not (os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT') or os.environ.get('GH_AW_OTLP_ENDPOINTS') or write_token):
        return
    try:
        logfire.configure(
            service_name=os.environ.get('OTEL_SERVICE_NAME', 'gh-aw'),
            send_to_logfire='if-token-present',
            console=False,
            token=write_token or None,
        )
        logfire.instrument_pydantic_ai(include_content=True, include_binary_content=True)
        logfire.instrument_httpx(capture_all=True)
        logfire.instrument_mcp()
        logger.info('Logfire/OTLP instrumentation enabled (pydantic_ai + httpx + mcp)')
    except Exception as exc:
        logger.warning('observability disabled: %r', exc)


def _mcp_tool_allowed(server: str, allowed: frozenset[str]) -> ToolPredicate:
    """Allow-list predicate matching gh-aw's `mcp__<server>__<tool>` form (or `mcp__<server>` wildcard)."""
    server_wildcard = f'mcp__{server}' in allowed

    def predicate(_ctx: RunContext[object], tool_def: ToolDefinition) -> bool:
        return server_wildcard or tool_def.name in allowed

    return predicate


def _apply_claude_mcp_prefix(entry: AbstractToolset[object]) -> AbstractToolset[object]:
    """Swap the default `<server>_<tool>` prefix for Claude Code's `mcp__<server>__<tool>` wire form.

    The trailing `_` combines with `PrefixedToolset`'s `_` separator to
    yield the doubled underscores gh-aw and Claude were trained on.
    """
    if not isinstance(entry, PrefixedToolset):
        return entry
    return dataclasses.replace(entry, prefix=f'mcp__{entry.prefix}_')


def build_mcp_servers(args: Args) -> list[AbstractToolset[object]]:
    """Load gh-aw's MCP config, re-prefix to Claude Code wire format, and apply the allow-list filter."""
    path = args.mcp_config or os.environ.get('GH_AW_MCP_CONFIG')
    if not path or not os.path.isfile(path):
        logger.info('no MCP config present — running without external tools')
        return []
    try:
        loaded = load_mcp_toolsets(path)
    # `repr` is sufficient diagnostically here (a `ValidationError` already
    # enumerates the bad fields, `FileNotFoundError` names the path), so no
    # traceback — but returning `[]` drops the *entire* GitHub/safeoutputs tool
    # surface, a drastic behaviour change, so log it at `error` to make a run
    # that silently lost its tools obvious in the artifact.
    except FileNotFoundError as exc:
        logger.error('MCP config %r missing (%r) — agent will run with NO external tools', path, exc)
        return []
    except (ValidationError, ValueError) as exc:
        logger.error('MCP config %r is malformed (%r) — agent will run with NO external tools', path, exc)
        return []

    servers: list[AbstractToolset[object]] = []
    for entry in loaded:
        name = (entry.wrapped.id if isinstance(entry, PrefixedToolset) else entry.id) or '<unnamed>'
        toolset = _apply_claude_mcp_prefix(cast('AbstractToolset[object]', entry))
        if args.allowed_tools is not None:
            toolset = toolset.filtered(_mcp_tool_allowed(name, args.allowed_tools))
            logger.info('registered MCP server %r (allow-list filtered)', name)
        else:
            logger.info('registered MCP server %r (no allow-list)', name)
        servers.append(toolset)
    return servers


def _claude_code_tool_predicate(allowed: frozenset[str] | None, permission_mode: str | None) -> ToolPredicate:
    """Allow-list + `plan`-mode filter for the Claude Code toolset."""
    plan = permission_mode == 'plan'

    def predicate(_ctx: RunContext[object], tool_def: ToolDefinition) -> bool:
        name = tool_def.name
        if allowed is not None and name not in allowed:
            return False
        if plan and name in MUTATING_TOOLS:
            return False
        return True

    return predicate


def select_claude_code_toolset(
    allowed: frozenset[str] | None,
    permission_mode: str | None,
    *,
    task: TaskCallable | None,
) -> AbstractToolset[object]:
    """Build the Claude Code toolset; `task=None` for sub-agents so they can't recurse."""
    return build_claude_code_toolset(task=task).filtered(_claude_code_tool_predicate(allowed, permission_mode))


# --------------------------------------------------------------------------- #
# Claude-compatible stream-json output
# --------------------------------------------------------------------------- #
def emit(obj: Mapping[str, object]) -> None:
    """Write one Claude-style stream-json line to stdout."""
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()


def emit_result(
    text: str,
    usage: RunUsage | None,
    session_id: str,
    is_error: bool = False,
    num_turns: int = 1,
    duration_ms: int = 0,
) -> None:
    """Emit the Claude Code stream-json `result` line gh-aw parses for success + token totals."""
    if usage is None:
        token_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cache_creation_input_tokens': 0,
            'cache_read_input_tokens': 0,
        }
    else:
        token_usage = {
            'input_tokens': usage.input_tokens,
            'output_tokens': usage.output_tokens,
            'cache_creation_input_tokens': usage.cache_write_tokens,
            'cache_read_input_tokens': usage.cache_read_tokens,
        }
    emit(
        {
            'type': 'result',
            'subtype': 'error' if is_error else 'success',
            'is_error': is_error,
            'result': text,
            'session_id': session_id,
            'num_turns': num_turns,
            'duration_ms': duration_ms,
            'total_cost_usd': 0,
            'usage': token_usage,
        }
    )


# Live tool-call / tool-result streaming for gh-aw's log parser. Result
# content is truncated for the stream view only — the model sees the full
# result via the message history.
MAX_LIVE_TOOL_RESULT_CHARS = 100


async def _stream_events(_ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
    """Emit tool_use / tool_result stream-json as events fire."""
    async for event in events:
        if isinstance(event, ToolCallEvent):
            emit(
                {
                    'type': 'assistant',
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'tool_use',
                                'id': event.part.tool_call_id,
                                'name': event.part.tool_name,
                                'input': event.part.args_as_dict(),
                            }
                        ],
                    },
                }
            )
            logger.info('tool_use: %s', event.part.tool_name)
        elif isinstance(event, ToolResultEvent):
            # `event.part` is `ToolReturnPart | RetryPromptPart`; the latter
            # means the tool result failed validation and pydantic-ai is
            # asking the model to retry. Tag it so gh-aw doesn't read it as
            # success.
            is_retry = isinstance(event.part, RetryPromptPart)
            content = str(event.part.content)
            if len(content) > MAX_LIVE_TOOL_RESULT_CHARS:
                content = (
                    content[:MAX_LIVE_TOOL_RESULT_CHARS] + f'…[+{len(content) - MAX_LIVE_TOOL_RESULT_CHARS} chars]'
                )
            emit(
                {
                    'type': 'user',
                    'message': {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'tool_result',
                                'tool_use_id': event.part.tool_call_id,
                                'content': content,
                                'is_error': is_retry,
                            }
                        ],
                    },
                }
            )


def count_tool_calls(messages: Sequence[ModelMessage]) -> int:
    """Tally tool calls in the final message history (for the end-of-run log)."""
    return sum(1 for m in messages for p in m.parts if isinstance(p, ToolCallPart))


def log_safe_outputs_state() -> None:
    """Log whether anything reached the gh-aw safe-outputs sink."""
    path = os.environ.get('GH_AW_SAFE_OUTPUTS')
    if not path:
        logger.info('GH_AW_SAFE_OUTPUTS not set')
        return
    try:
        data = pathlib.Path(path).read_text(encoding='utf-8')
    except OSError as exc:
        logger.info('GH_AW_SAFE_OUTPUTS unreadable (%s): %r', path, exc)
        return
    lines = [ln for ln in data.splitlines() if ln.strip()]
    logger.info('GH_AW_SAFE_OUTPUTS=%s entries=%d bytes=%d', path, len(lines), len(data))
    for ln in lines[:5]:
        logger.info('  safe-output: %s', ln[:300])


async def task(ctx: RunContext[object], description: str, prompt: str) -> str:
    """Claude's `Task` tool: spawn a read-only sub-agent on `ctx.model`."""
    logger.info('Task spawn: %s', description[:120])
    # Fresh dedupe set per sub-agent — otherwise inheriting the parent's
    # `seen` AGENTS.md set would silently hide context the sub-agent needs.
    reset_context_state()
    sub_toolset = select_claude_code_toolset(READ_ONLY_SUBAGENT_TOOLS, permission_mode=None, task=None)
    sub = Agent(
        ctx.model,
        instructions=[INSTRUCTIONS, SUBAGENT_INSTRUCTIONS, prompt],
        toolsets=[sub_toolset],
        capabilities=[
            *_anthropic_native_capabilities(),
            ProcessEventStream(_stream_events),
        ],
    )
    # Fresh `RunUsage` so `SUBAGENT_REQUEST_LIMIT` bounds the sub-agent, not
    # (parent + sub). Merge the deltas back regardless of success/failure.
    sub_usage = RunUsage()
    try:
        result = await asyncio.wait_for(
            sub.run(RUN_TRIGGER, usage_limits=UsageLimits(request_limit=SUBAGENT_REQUEST_LIMIT), usage=sub_usage),
            timeout=SUBAGENT_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        # A bare `TimeoutError` stringifies to '' — without an explicit message
        # the model (and the log) would see `sub-agent failed:` with no payload.
        ctx.usage.incr(sub_usage)
        logger.error('sub-agent timed out after %.0f min: %s', SUBAGENT_TIMEOUT_SECS / 60, description[:120])
        return f'error: sub-agent timed out after {SUBAGENT_TIMEOUT_SECS // 60}min'
    except Exception as exc:
        # The parent agent reacts to the returned string, but a sub-agent can hit
        # the same nested `ExceptionGroup`/`McpError` as the main run — log the
        # full stack so the failure isn't reduced to a one-line repr in the logs.
        ctx.usage.incr(sub_usage)
        logger.exception('sub-agent failed: %s', description[:120])
        return f'error: sub-agent failed: {exc}'
    ctx.usage.incr(sub_usage)
    logger.info('Task done: +%d sub-requests (run total now %d)', sub_usage.requests, ctx.usage.requests)
    return str(result.output or '')


async def _run_with_timeout(
    prompt: str,
    model: Model,
    label: str,
    claude_code_toolset: AbstractToolset[object],
    mcp_servers: list[AbstractToolset[object]],
    session_id: str,
) -> int:
    """Wrap `run()` with the global wall-clock cap and emit a clean result on timeout."""
    try:
        return await asyncio.wait_for(
            run(prompt, model, label, claude_code_toolset, mcp_servers, session_id),
            timeout=RUN_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        logger.error('run timed out after %.0f min', RUN_TIMEOUT_SECS / 60)
        emit_result(
            f'run timed out after {RUN_TIMEOUT_SECS // 60}min',
            usage=None,
            session_id=session_id,
            is_error=True,
        )
        return 1


async def run(
    prompt: str,
    model: Model,
    label: str,
    claude_code_toolset: AbstractToolset[object],
    mcp_servers: list[AbstractToolset[object]],
    session_id: str,
) -> int:
    """Run one agent turn and emit Claude-shape stream-json. Always emits a `result` line."""
    reset_context_state()
    agent: Agent[object, str] = Agent(
        model,
        instructions=[INSTRUCTIONS, prompt],
        toolsets=[claude_code_toolset, *mcp_servers],
        capabilities=[
            *_anthropic_native_capabilities(),
            _history_compaction(),
            ProcessEventStream(_stream_events),
        ],
    )
    limits = UsageLimits(request_limit=REQUEST_LIMIT)
    emit({'type': 'system', 'subtype': 'init', 'session_id': session_id, 'model': label})

    started = time.perf_counter()
    try:
        async with agent:
            result = await agent.run(RUN_TRIGGER, usage_limits=limits)
    except Exception as exc:
        # `%r` on an `ExceptionGroup` (e.g. the MCP `TaskGroup` failures seen in
        # CI) discards every frame and every nested sub-exception's stack, which
        # is what made the original incident so hard to root-cause. `exception()`
        # renders the full traceback — and, on 3.11+, each group leaf's stack —
        # to stderr, which gh-aw captures into the uploaded `agent-stdio.log`,
        # so the run is self-explaining without a re-run. The `result` text
        # stays a one-liner because gh-aw parses it.
        logger.exception('agent run failed')
        emit_result(
            f'agent run failed: {exc}',
            usage=None,
            session_id=session_id,
            is_error=True,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        return 1

    duration_ms = round((time.perf_counter() - started) * 1000)
    messages = result.all_messages()
    tool_calls = count_tool_calls(messages)
    num_turns = sum(isinstance(m, ModelResponse) for m in messages)
    logger.info('tool calls observed: %d, turns: %d', tool_calls, num_turns)

    text = str(result.output or '')
    emit({'type': 'assistant', 'message': {'role': 'assistant', 'content': text}})

    emit_result(text, result.usage, session_id, num_turns=num_turns, duration_ms=duration_ms)
    log_safe_outputs_state()
    return 0


def main() -> int:
    """Entry point. Every failure produces a stream-json `result` line so gh-aw never sees an empty log."""
    configure_logging()
    session_id = (os.environ.get('GITHUB_RUN_ID') or 'local') + '-' + uuid.uuid4().hex[:8]
    try:
        args = parse_args(sys.argv[1:])
        configure_observability()
        prompt = resolve_prompt(args)
        if not prompt.strip():
            logger.info('empty prompt — nothing to do')
            emit_result('empty prompt', usage=None, session_id=session_id, is_error=True)
            return 1
        model, label = build_model(args)
        claude_code_toolset = select_claude_code_toolset(args.allowed_tools, args.permission_mode, task=task)
        mcp_servers = build_mcp_servers(args)
        logger.info(
            'model=%s permission_mode=%s request_limit=%d claude_code_tool_names=%s mcp_servers=%d prompt_chars=%d',
            label,
            args.permission_mode or '(none)',
            REQUEST_LIMIT,
            list(CLAUDE_CODE_TOOL_NAMES),
            len(mcp_servers),
            len(prompt),
        )
        started = time.time()
        rc = asyncio.run(_run_with_timeout(prompt, model, label, claude_code_toolset, mcp_servers, session_id))
        logger.info('done in %.1fs rc=%d', time.time() - started, rc)
        return rc
    except SystemExit as exc:
        # `argparse` raises `SystemExit` (not `Exception`) on unknown-flag
        # rejection — an expected, clean exit, so a traceback would be noise.
        # gh-aw still needs a structured result line.
        logger.error('FATAL startup error: %r', exc)
        emit_result(f'shim startup failed: {exc}', usage=None, session_id=session_id, is_error=True)
        return 1
    except Exception as exc:
        # A real crash before the agent run (model build, MCP load, …) — dump the
        # full stack so a blind FATAL doesn't cost another long investigation.
        logger.exception('FATAL startup error')
        emit_result(f'shim startup failed: {exc}', usage=None, session_id=session_id, is_error=True)
        return 1
