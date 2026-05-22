r"""Pydantic AI gh-aw shim — Claude Code CLI compatibility for gh-aw.

gh-aw runs the agent engine like the Claude Code CLI:

    <command> --print --no-chrome --allowed-tools '<csv>' --debug-file <path> \\
      --verbose --permission-mode <mode> --output-format stream-json \\
      --mcp-config <mcp-servers.json> --prompt-file <prompt.txt> \\
      [--model <model>] "<rendered prompt>"

With `engine.command` set, `<command>` is this shim. It speaks Claude
Code's argv, recovers the prompt, builds a `pydantic-ai` agent backed by
the gh-aw-injected Anthropic-compatible proxy, exposes Claude-named native
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
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import logfire
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeTool, ProcessEventStream, ProcessHistory
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolSearchCallPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultEvent,
    ToolReturnPart,
    ToolSearchCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.native_tools import WebFetchTool
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from . import (
    MUTATING_TOOLS,
    NATIVE_TOOL_NAMES,
    READ_ONLY_SUBAGENT_TOOLS,
    build_native_toolset,
)
from .shared import logger, reset_context_state

# Type aliases for the public surface — the shim runs `None`-deps agents
# throughout, so every `RunContext` is concretely `RunContext[None]`.
MessagePart: TypeAlias = ModelRequestPart | ModelResponsePart
ToolPredicate: TypeAlias = Callable[[RunContext[None], ToolDefinition], bool | Awaitable[bool]]
TaskCallable: TypeAlias = Callable[[RunContext[None], str, str], Awaitable[str]]

# Placeholder bearer token sent to the AWF api-proxy. The proxy strips this
# header and injects the real `ANTHROPIC_API_KEY` on the outbound wire — so
# the agent container never sees the real key. Sent verbatim only when no
# `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` env is provided locally.
PROXY_BEARER_PLACEHOLDER = 'gh-aw-proxy-injected'

# Cap on model requests for the parent agent and each `Task` sub-agent.
# The Claude Code CLI the shim replaces has no such cap; pydantic-ai's
# built-in default is only 50, which deep multi-step agents exhaust
# mid-task. gh-aw's api-proxy still enforces the real backstop
# (maxRuns/token budget).
REQUEST_LIMIT = 200
SUBAGENT_REQUEST_LIMIT = 75

# Execution guidance that prepends every workflow prompt as a system
# instruction. Weaker models (e.g. MiniMax behind an Anthropic-compatible
# shim) follow system instructions much more strictly than user messages,
# so `run()` and `task()` pass this as the first element of
# `Agent(instructions=[INSTRUCTIONS, prompt])` — the sequence form keeps
# this static prefix stable across runs so Anthropic prompt caching can
# hit it. The user message is the trivial `RUN_TRIGGER` below;
# pydantic-ai re-applies `instructions` to every `ModelRequest`, so the
# task spec stays in view across the entire run (and across compaction).
INSTRUCTIONS = (
    'You can call multiple tools in a single turn. When tool calls are '
    "independent (no call needs another call's output), issue them together "
    'in one turn so they run in parallel — this is faster and uses far fewer '
    'model requests. Only chain tools sequentially when one genuinely needs a '
    "previous tool's result."
)

# Trivial user message that triggers the agent to start working. The actual
# task spec lives in `instructions=[INSTRUCTIONS, prompt]`, so the user
# message is just a "begin" signal.
RUN_TRIGGER = 'Begin the task per the instructions above.'

SUBAGENT_INSTRUCTIONS = (
    'You are a focused, read-only sub-agent. You can read files, search the '
    'codebase, and fetch web content, but you cannot modify the workspace or '
    'shell out. Investigate the task you were given and return a concise, '
    'evidence-grounded answer to your caller — do not try to act on it.'
)


# --------------------------------------------------------------------------- #
# History compaction (pydantic-ai `ProcessHistory` capability)
#
# Long agent runs (deep sweeps, large-diff PR reviews) accumulate huge
# message histories. We bound that growth in two stages — a cheap
# non-destructive trim that almost always suffices, then an LLM summary as
# the fallback. Both run inside one `ProcessHistory` callback.
#
# The workflow prompt + `INSTRUCTIONS` ride on `Agent(instructions=...)`,
# which pydantic-ai re-applies on every `ModelRequest`. They never enter
# the message list and so are *never* touched by either stage.
# --------------------------------------------------------------------------- #

# ~100k tokens at 4 chars/token = half of a 200k window. Past this the summariser fires.
COMPACTION_TRIGGER_CHARS = 400_000
COMPACTION_KEEP_RECENT = 10  # last N messages always preserved verbatim
TOOL_RESULT_HEAD_TAIL_CHARS = 800  # head + tail each kept when trimming an older tool result
TOOL_RESULT_TRIM_THRESHOLD = 2_000  # tool result above this gets head/tail trimmed
COMPACTION_TRANSCRIPT_MAX_CHARS = 80_000  # max input shipped to the summariser

# Structured-section summary prompt. Borrowed shape from Anthropic's
# compaction guidance, OpenHands' condenser, and Cursor Composer's
# self-summarisation: named sections at a generous (2–4k token) budget beat
# free prose at a tight one. The `do not call tools` line is defensive —
# the summariser sub-agent has no tools today, so this guards against any
# future regression that wires them in.
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
    'Respond with text only — do not call any tools.'
)


def _part_text(part: MessagePart) -> str:
    """Best-effort text for any pydantic-ai message part.

    Two shapes exist across the message union: tool-call parts expose
    `tool_name` + `args`, every other part exposes `content` (str or a
    structured object that stringifies usefully). Splitting on the call-part
    isinstance keeps the call site fully type-checked without a fallback
    branch that would mask new part subclasses pydantic-ai introduces.
    """
    if isinstance(part, (ToolCallPart, NativeToolCallPart, ToolSearchCallPart, NativeToolSearchCallPart)):
        return part.tool_name
    return str(part.content)


def _render_messages_for_summary(messages: list[ModelMessage]) -> str:
    """Render a slice of pydantic-ai messages into a compact transcript."""
    out: list[str] = []
    for m in messages:
        kind = 'user' if isinstance(m, ModelRequest) else 'assistant'
        for part in m.parts:
            out.append(f'[{kind}/{type(part).__name__}] {_part_text(part)[:1500]}')
    return '\n'.join(out)


def _history_size_chars(messages: list[ModelMessage]) -> int:
    """Rough char budget for a history (sum of every text-bearing part).

    A cheap, model-free proxy for token cost (~3-4 chars per token). Used to
    decide when to compact; precise tokenisation isn't worth the cost or the
    per-provider tokenizer dependency.
    """
    return sum(len(_part_text(part)) for m in messages for part in m.parts)


def _head_tail(text: str, side: int) -> str:
    """Keep the first and last `side` chars, mark the elided middle."""
    skipped = len(text) - side * 2
    return f'{text[:side]}\n…[trimmed {skipped} chars]…\n{text[-side:]}'


def _trim_tool_results(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Cheap, non-destructive pre-pass that runs before any LLM summary.

    Two transformations on tool results that live *before* the last
    `KEEP_RECENT` messages (recent context is always preserved exactly):

    1. **Dedupe superseded file reads.** When `Read(file_path=X)` is called
       more than once, every older read result for `X` is replaced with a
       single-line marker pointing at the most recent one. Re-reads are by
       far the largest source of redundant transcript bytes in coding
       agents (Cline reports 30–50%); they carry no information beyond
       what the latest read already shows.

    2. **Head/tail-truncate oversized tool results.** Tool returns larger
       than `TOOL_RESULT_TRIM_THRESHOLD` keep the first + last
       `TOOL_RESULT_HEAD_TAIL_CHARS` and elide the middle.

    Often enough on its own to skip the summariser entirely.
    """
    if len(messages) <= COMPACTION_KEEP_RECENT:
        return messages

    # Pass 1: walk forward, record file_path -> most-recent Read call_id. Any
    # earlier call_id for the same path is marked superseded.
    read_path_by_call_id: dict[str, str] = {}
    latest_read_for_path: dict[str, str] = {}
    superseded: set[str] = set()
    for m in messages:
        for p in m.parts:
            if isinstance(p, ToolCallPart) and p.tool_name == 'Read':
                args = p.args_as_dict()
                path = args.get('file_path') if isinstance(args, dict) else None
                if isinstance(path, str):
                    read_path_by_call_id[p.tool_call_id] = path
                    prior = latest_read_for_path.get(path)
                    if prior is not None:
                        superseded.add(prior)
                    latest_read_for_path[path] = p.tool_call_id

    # Pass 2: rebuild older messages with substitutions. Tail (KEEP_RECENT)
    # passes through verbatim. Return the original list object when no
    # substitution fired — preserves identity for the no-op fast path.
    tail_start = len(messages) - COMPACTION_KEEP_RECENT
    out: list[ModelMessage] = []
    dedup_count = 0
    truncate_count = 0
    bytes_saved = 0
    for idx, m in enumerate(messages):
        if idx >= tail_start:
            out.append(m)
            continue
        new_parts: list[ModelRequestPart | ModelResponsePart] = []
        msg_changed = False
        for part in m.parts:
            if isinstance(part, ToolReturnPart):
                if part.tool_call_id in superseded:
                    path = read_path_by_call_id[part.tool_call_id]
                    new_content = f'[superseded read: {path} — see later read]'
                    bytes_saved += len(str(part.content)) - len(new_content)
                    dedup_count += 1
                    new_parts.append(dataclasses.replace(part, content=new_content))
                    msg_changed = True
                    continue
                content = str(part.content)
                if len(content) > TOOL_RESULT_TRIM_THRESHOLD:
                    new_content = _head_tail(content, TOOL_RESULT_HEAD_TAIL_CHARS)
                    bytes_saved += len(content) - len(new_content)
                    truncate_count += 1
                    new_parts.append(dataclasses.replace(part, content=new_content))
                    msg_changed = True
                    continue
            new_parts.append(part)
        if msg_changed:
            out.append(dataclasses.replace(m, parts=new_parts))
        else:
            out.append(m)
    if dedup_count or truncate_count:
        logger.info(
            'compaction trim: deduped %d superseded read(s), truncated %d oversized result(s), saved %d chars',
            dedup_count,
            truncate_count,
            bytes_saved,
        )
        return out
    return messages


async def _compact_history(ctx: RunContext[None], messages: list[ModelMessage]) -> list[ModelMessage]:
    """Two-stage history compaction — cheap trim first, LLM summary as fallback.

    Stage 1 — `_trim_tool_results` dedupes superseded file reads and
    head/tail-truncates oversized tool returns. Non-destructive: nothing
    informative is dropped.

    Stage 2 — only if the trimmed history is *still* over budget, summarise
    everything before the last `KEEP_RECENT` messages into a structured
    brief (`COMPACTION_SUMMARY_INSTRUCTIONS`). On summariser failure, fall
    back to plain truncation.
    """
    if len(messages) <= COMPACTION_KEEP_RECENT:
        return messages
    trimmed = _trim_tool_results(messages)
    size = _history_size_chars(trimmed)
    if size < COMPACTION_TRIGGER_CHARS:
        return trimmed
    middle = trimmed[:-COMPACTION_KEEP_RECENT]
    tail = trimmed[-COMPACTION_KEEP_RECENT:]
    transcript = _render_messages_for_summary(middle)
    logger.info(
        'compaction summary firing: %d chars across %d messages exceeds %d-char trigger; '
        'summarising %d middle messages, keeping last %d verbatim',
        size,
        len(trimmed),
        COMPACTION_TRIGGER_CHARS,
        len(middle),
        COMPACTION_KEEP_RECENT,
    )
    try:
        summariser = Agent(ctx.model, instructions=COMPACTION_SUMMARY_INSTRUCTIONS)
        # Share the parent's RunUsage so the summariser's tokens roll up into
        # the run's reported totals.
        r = await summariser.run(
            f'Transcript to summarise:\n\n{transcript[:COMPACTION_TRANSCRIPT_MAX_CHARS]}',
            usage_limits=UsageLimits(request_limit=2),
            usage=ctx.usage,
        )
        summary = str(r.output or '').strip() or '(empty summary)'
    except Exception as exc:
        # Any failure (network, model error, usage-limit, content filter) must
        # degrade to plain truncation rather than kill the parent run.
        logger.warning('compaction summarisation failed (%r); truncating instead', exc)
        return tail
    logger.info(
        'compaction summary done: replaced %d middle messages with a %d-char structured summary',
        len(middle),
        len(summary),
    )
    synthetic = ModelRequest(parts=[UserPromptPart(content=f'[compacted history]\n{summary}')])
    return [synthetic, *tail]


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
    client = AsyncAnthropic(auth_token=auth_token, base_url=anthropic_base)
    return (
        AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client)),
        f'anthropic:{model_name}',
    )


def configure_logging() -> None:
    """Configure stderr logging once, at CLI entry.

    Library code (the
    `shared` module and tools) only owns the named logger; calling
    `basicConfig` at import time would clobber the embedder's logging
    setup (and trip the test runner).
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[pydantic-ai-gh-aw-shim] %(message)s',
        stream=sys.stderr,
    )


def configure_observability() -> None:
    """pydantic-ai is natively instrumented; gh-aw injects OTEL_*/LOGFIRE_TOKEN so traces flow with no extra config.

    Never let observability break the run.

    """
    if not (
        os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
        or os.environ.get('GH_AW_OTLP_ENDPOINTS')
        or os.environ.get('LOGFIRE_TOKEN')
    ):
        return
    try:
        logfire.configure(
            service_name=os.environ.get('OTEL_SERVICE_NAME', 'gh-aw'),
            send_to_logfire='if-token-present',
            console=False,
        )
        # Capture full payloads (prompts, tool args/results, HTTP bodies)
        # so the trace is debuggable end-to-end. The agent runs in the
        # AWF sandbox; these spans land in Logfire/OTLP and never on
        # disk in the shim.
        logfire.instrument_pydantic_ai(include_content=True, include_binary_content=True)
        logfire.instrument_httpx(capture_all=True)
        logfire.instrument_mcp()
        logger.info('Logfire/OTLP instrumentation enabled (pydantic_ai + httpx + mcp)')
    except Exception as exc:
        # Instrumentation setup touches the network/auth and several optional
        # libs; any failure here must degrade silently so the agent run still
        # proceeds without telemetry.
        logger.warning('observability disabled: %r', exc)


# --------------------------------------------------------------------------- #
# MCP bridge
# --------------------------------------------------------------------------- #
def _mcp_tool_allowed(server: str, allowed: frozenset[str]) -> ToolPredicate:
    """Build a `MCPToolset.filtered` predicate enforcing gh-aw's allow-list.

    The model-visible tool name is `mcp__<server>__<tool>` (Claude Code's
    wire format — see `_apply_claude_mcp_prefix`), so this is a literal
    containment check against gh-aw's allow-list, with a wildcard for
    whole-server entries.
    """
    server_wildcard = f'mcp__{server}' in allowed

    def predicate(_ctx: RunContext[None], tool_def: ToolDefinition) -> bool:
        return server_wildcard or tool_def.name in allowed

    return predicate


def _apply_claude_mcp_prefix(entry: AbstractToolset[None]) -> AbstractToolset[None]:
    """Re-prefix a `load_mcp_toolsets` result to Claude Code's wire format.

    `load_mcp_toolsets` returns each server wrapped in
    `PrefixedToolset(prefix=server)` so the model sees `<server>_<tool>`.
    Claude Code (and gh-aw's allow-list) uses `mcp__<server>__<tool>` for
    every MCP tool — that's the name Claude was trained on and the name
    workflow authors write in `--allowed-tools`. We swap the prefix to
    `mcp__<server>_` (with the trailing underscore baked in) so that
    `PrefixedToolset`'s hardcoded `_` separator produces `__` between
    prefix and tool name, yielding the canonical Claude Code shape.
    """
    if not isinstance(entry, PrefixedToolset):
        return entry
    return dataclasses.replace(entry, prefix=f'mcp__{entry.prefix}_')


def build_mcp_servers(args: Args) -> list[AbstractToolset[None]]:
    """Translate gh-aw's Claude-format `mcp-servers.json` into MCP toolsets.

    Delegates parsing + transport selection + `${VAR}`/`${VAR:-default}` env
    expansion to `pydantic_ai.mcp.load_mcp_toolsets`, then re-prefixes each
    server to Claude Code's wire format (`mcp__<server>__<tool>`) so the
    model sees the same identifier that gh-aw writes into `--allowed-tools`
    and that Claude was trained on. With matching names, the allow-list
    filter is a literal containment check.
    """
    path = args.mcp_config or os.environ.get('GH_AW_MCP_CONFIG')
    if not path or not os.path.isfile(path):
        logger.info('no MCP config present — running without external tools')
        return []
    try:
        loaded = load_mcp_toolsets(path)
    except FileNotFoundError as exc:
        logger.warning('MCP config %r missing: %r — running without external tools', path, exc)
        return []
    except (ValidationError, ValueError) as exc:
        # `ValidationError`: schema mismatch (unknown server type, missing fields).
        # `ValueError`: an `${UNDEFINED_VAR}` reference with no `:-default`.
        logger.warning('MCP config %r is malformed: %r — running without external tools', path, exc)
        return []

    servers: list[AbstractToolset[None]] = []
    for entry in loaded:
        # Pull the server name off the inner `MCPToolset` (which `load_mcp_toolsets`
        # initialised from the `mcpServers` map key) before we re-prefix.
        name = (entry.wrapped.id if isinstance(entry, PrefixedToolset) else entry.id) or '<unnamed>'
        # `load_mcp_toolsets` returns `AbstractToolset[Any]`; the shim's agents
        # are all `None`-deps, so this cast tightens the surface honestly.
        toolset = _apply_claude_mcp_prefix(cast('AbstractToolset[None]', entry))
        if args.allowed_tools is not None:
            toolset = toolset.filtered(_mcp_tool_allowed(name, args.allowed_tools))
            logger.info('registered MCP server %r (allow-list filtered)', name)
        else:
            logger.info('registered MCP server %r (no allow-list)', name)
        servers.append(toolset)
    return servers


# --------------------------------------------------------------------------- #
# Native toolset selection
# --------------------------------------------------------------------------- #
def _native_tool_predicate(allowed: frozenset[str] | None, permission_mode: str | None) -> ToolPredicate:
    """Build a `.filtered()` predicate enforcing gh-aw's allow-list + plan mode for the native toolset.

    `allowed=None` (flag absent) keeps every tool. `plan` mode withholds
    workspace-mutating tools.
    """
    plan = permission_mode == 'plan'

    def predicate(_ctx: RunContext[None], tool_def: ToolDefinition) -> bool:
        name = tool_def.name
        if allowed is not None and name not in allowed:
            return False
        if plan and name in MUTATING_TOOLS:
            return False
        return True

    return predicate


def select_native_toolset(
    allowed: frozenset[str] | None,
    permission_mode: str | None,
    *,
    task: TaskCallable | None,
) -> AbstractToolset[None]:
    """Build the parent or sub-agent native toolset.

    Pass `task=` only for
    the parent; sub-agents get `task=None` so they can't spawn further
    sub-agents. Returns `AbstractToolset[None]` because `.filtered(...)`
    yields a `FilteredToolset`, not a `FunctionToolset`.
    """
    return build_native_toolset(task=task).filtered(_native_tool_predicate(allowed, permission_mode))


# --------------------------------------------------------------------------- #
# Claude-compatible stream-json output
# --------------------------------------------------------------------------- #
def emit(obj: Mapping[str, object]) -> None:
    """Write one Claude-style stream-json line to stdout.

    Callers pass a literal dict whose nested values are JSON-serialisable
    primitives; `Mapping[str, object]` captures that shape without leaking
    `Any` into the public signature.
    """
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
    """Emit Claude Code's stream-json `result` line — gh-aw parses this to decide success/failure and to populate token-count rollups.

    `usage=None`
    is used on the error/startup path where there's no run to count.

    """
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


# --------------------------------------------------------------------------- #
# Live tool-call / tool-result streaming
# Wired in as `ProcessEventStream` on both the parent agent and each `Task`
# sub-agent so events fire as they happen, not in a batch at end-of-run.
# Sub-agent events naturally interleave with the parent's because they're
# spawned inside the parent's `Task` tool execution, so the stream order is
# the true causal order. gh-aw's log parser shows each tool call live in the
# step summary; tool results are truncated to MAX_LIVE_TOOL_RESULT_CHARS to
# match other gh-aw engines (Claude, Copilot) — the model still sees the
# full result via the message history.
# --------------------------------------------------------------------------- #
MAX_LIVE_TOOL_RESULT_CHARS = 100


async def _stream_events(_ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
    """ProcessEventStream handler — emit tool_use / tool_result stream-json blocks as they happen.

    Truncates result content for visibility; the
    model's view is unaffected (the handler is observation-only).

    """
    async for event in events:
        # Use the `ToolCallEvent` / `ToolResultEvent` base classes so we
        # cover both `FunctionToolCallEvent` (the common path) and
        # `OutputToolCallEvent` (pydantic-ai structured-output mechanism)
        # — they share the `part` attribute we read from.
        if isinstance(event, ToolCallEvent):
            emit(
                {
                    'type': 'assistant',
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'tool_use',
                                'id': event.part.tool_call_id or '',
                                'name': event.part.tool_name or '',
                                'input': event.part.args_as_dict(),
                            }
                        ],
                    },
                }
            )
            logger.info('tool_use: %s', event.part.tool_name)
        elif isinstance(event, ToolResultEvent):
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
                                'tool_use_id': event.part.tool_call_id or '',
                                'content': content,
                            }
                        ],
                    },
                }
            )


def count_tool_calls(messages: Sequence[ModelMessage]) -> int:
    """Tally tool calls in the final message history for the end-of-run log line.

    Live emission is handled by `_stream_events`, so we no longer
    re-emit the full transcript here.

    """
    return sum(1 for m in messages for p in m.parts if isinstance(p, ToolCallPart))


def log_safe_outputs_state() -> None:
    """Log whether anything reached the gh-aw safe-outputs sink — that file gates the downstream safe_outputs job."""
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


# --------------------------------------------------------------------------- #
# Sub-agent dispatcher (Claude's `Task` tool)
# Defined here (not in the package) because it needs the parent-shim
# `Agent` factory, prompt constants, and event-stream handler.
# --------------------------------------------------------------------------- #
async def task(ctx: RunContext[None], description: str, prompt: str) -> str:
    """Dispatch a focused **read-only sub-agent** (Claude's `Task` tool).

    The sub-agent runs on the same model as the caller (via `ctx.model`),
    with only the read-only native tools (`READ_ONLY_SUBAGENT_TOOLS`) — no
    Bash, Write, Edit, MultiEdit, or nested Task. Used by orchestrating
    prompts to parallelise investigation (e.g. one sub-agent per file in a
    PR review) and then merge the findings.
    """
    logger.info('Task spawn: %s', description[:120])
    sub_toolset = select_native_toolset(READ_ONLY_SUBAGENT_TOOLS, permission_mode=None, task=None)
    # Same wiring as the parent in `run()`: the (typically long, structured)
    # task prompt rides as a system instruction so weaker models follow it
    # strictly. Parent agents are instructed to build fully-self-contained
    # sub-agent prompts (full PR context, severity scale, file list, output
    # format) — that material belongs in the system role, not as a user
    # message. The user message is the trivial `RUN_TRIGGER`.
    sub = Agent(
        ctx.model,
        # Sequence form keeps the static `INSTRUCTIONS` / `SUBAGENT_INSTRUCTIONS`
        # prefix stable across runs so Anthropic prompt caching can hit it.
        # F-string concatenation would change every prompt and bust the cache.
        instructions=[INSTRUCTIONS, SUBAGENT_INSTRUCTIONS, prompt],
        toolsets=[sub_toolset],
        capabilities=[
            NativeTool(WebFetchTool()),
            ProcessEventStream(_stream_events),
        ],
    )
    limits = UsageLimits(request_limit=SUBAGENT_REQUEST_LIMIT)
    # Share the parent's RunUsage so the sub-agent's tokens/requests/tool-calls
    # roll up into the run's final stream-json result — otherwise sub-agent
    # cost is invisible.
    before_requests = ctx.usage.requests
    try:
        result = await sub.run(RUN_TRIGGER, usage_limits=limits, usage=ctx.usage)
    except Exception as exc:
        # Sub-agent failures (model errors, usage-limit breaches, tool
        # exceptions) surface here as a tool-result string the parent agent
        # can react to, rather than crashing the whole orchestrator run.
        return f'error: sub-agent failed: {exc}'
    logger.info(
        'Task done: +%d sub-requests (total=%d)',
        ctx.usage.requests - before_requests,
        ctx.usage.requests,
    )
    return str(result.output or '')


async def run(
    prompt: str,
    model: Model,
    label: str,
    native_toolset: AbstractToolset[None],
    mcp_servers: list[AbstractToolset[None]],
    session_id: str,
) -> int:
    """Run a single agent turn end-to-end and emit Claude-compatible stream-json on stdout.

    Always emits a `result` line, success or
    failure.

    """
    reset_context_state()  # per-run dedupe state for AGENTS.md/CLAUDE.md auto-loading
    # The workflow prompt rides as a system instruction (not a user message)
    # because weaker models follow system prompts much more strictly than
    # user prompts — and the workflow prompts here are full task specs, not
    # ad-hoc questions. The user message is a trivial trigger.
    agent: Agent[None, str] = Agent(
        model,
        # Sequence form keeps the static `INSTRUCTIONS` prefix cacheable by
        # Anthropic's prompt-prefix cache; the workflow `prompt` varies per
        # run but the leading element doesn't, so the cache hits the prefix.
        instructions=[INSTRUCTIONS, prompt],
        toolsets=[native_toolset, *mcp_servers],
        capabilities=[
            NativeTool(WebFetchTool()),
            ProcessHistory(_compact_history),
            ProcessEventStream(_stream_events),
        ],
    )
    limits = UsageLimits(request_limit=REQUEST_LIMIT)
    emit({'type': 'system', 'subtype': 'init', 'session_id': session_id, 'model': label})

    started = time.perf_counter()
    try:
        # Always enter the agent context — `Agent.__aenter__` is a cheap no-op
        # without MCP toolsets, and using it unconditionally means any future
        # capability or toolset that needs lifecycle management is handled
        # correctly without revisiting this branch. `logfire.instrument_mcp()`
        # (set up in `configure_observability()`) already traces MCP tool
        # discovery and invocations, so we don't add diagnostic logging here.
        async with agent:
            result = await agent.run(RUN_TRIGGER, usage_limits=limits)
    except Exception as exc:
        # provider errors, usage-limit breaches, tool exceptions, or
        # transport timeouts; gh-aw needs a structured `result` line for
        # every outcome, so we catch broadly and re-emit as is_error=True.
        logger.warning('agent run failed: %r', exc)
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
    """Entry point.

    Any failure is surfaced as a Claude-style stream-json error
    result so gh-aw always has a structured outcome to parse (never an opaque
    'no structured log entries' run).
    """
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
        native_toolset = select_native_toolset(args.allowed_tools, args.permission_mode, task=task)
        mcp_servers = build_mcp_servers(args)
        logger.info(
            'model=%s permission_mode=%s request_limit=%d native_tool_names=%s mcp_servers=%d prompt_chars=%d',
            label,
            args.permission_mode or '(none)',
            REQUEST_LIMIT,
            list(NATIVE_TOOL_NAMES),
            len(mcp_servers),
            len(prompt),
        )
        started = time.time()
        rc = asyncio.run(run(prompt, model, label, native_toolset, mcp_servers, session_id))
        logger.info('done in %.1fs rc=%d', time.time() - started, rc)
        return rc
    except Exception as exc:
        # Startup-phase failures (argv, MCP-config parse, model construction,
        # network) must still produce a Claude-shape `result` line — gh-aw
        # treats an absent result as an opaque infra failure rather than an
        # agent error.
        logger.error('FATAL startup error: %r', exc)
        emit_result(f'shim startup failed: {exc}', usage=None, session_id=session_id, is_error=True)
        return 1
