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
(`ANTHROPIC_BASE_URL` → Anthropic or another Anthropic-compatible endpoint).
No OpenAI path — the workflow's `engine.id: claude` contract is
Anthropic-shape end to end.

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
import math
import os
import pathlib
import sys
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterable, Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Annotated, Any, TypeAlias, cast

import httpx2
import logfire
from anthropic import APITimeoutError, AsyncAnthropic
from mcp.shared.exceptions import McpError
from opentelemetry.trace import Span, Status, StatusCode, get_current_span
from pydantic import Field, TypeAdapter, ValidationError
from tenacity import RetryCallState, stop_after_delay, wait_random_exponential
from tenacity.asyncio.retry import retry_if_result

from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelRequestContext,
    ModelRetry,
    RunContext,
    UsageLimitExceeded,
    capture_run_messages,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    NativeTool,
    PrepareTools,
    ProcessEventStream,
    ProcessHistory,
    WrapModelRequestHandler,
    WrapRunHandler,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.mcp import load_mcp_toolsets
from pydantic_ai.messages import (
    AgentStreamEvent,
    InstructionBaselinePart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolSearchCallPart,
    ToolAvailabilityDeltaPart,
    ToolCallEvent,
    ToolCallPart,
    ToolResultEvent,
    ToolReturnPart,
    ToolSearchCallPart,
    UserPromptPart,
    repair_messages,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.native_tools import WebFetchTool
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, PrefixedToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from . import (
    CLAUDE_CODE_TOOL_NAMES,
    MUTATING_TOOLS,
    READ_ONLY_SUBAGENT_TOOLS,
    build_claude_code_toolset,
)
from ._backends import local_workspace
from .shared import logger, reset_context_state

# Type aliases for the public surface — the shim runs `None`-deps agents
# throughout, so every `RunContext` is concretely `RunContext[object]`.
MessagePart: TypeAlias = ModelRequestPart | ModelResponsePart
ToolPredicate: TypeAlias = Callable[[RunContext[object], ToolDefinition], bool | Awaitable[bool]]
TaskCallable: TypeAlias = Callable[[RunContext[object], str, str], Awaitable[str]]

# Placeholder bearer token sent to the AWF api-proxy. The proxy strips this
# header and injects the real `ANTHROPIC_API_KEY` on the outbound wire — so
# the agent container never sees the real key. Sent verbatim only when no
# `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_API_KEY` env is provided locally.
PROXY_BEARER_PLACEHOLDER = 'gh-aw-proxy-injected'


def _anthropic_native_capabilities() -> list[NativeTool]:
    """`NativeTool(WebFetchTool())` for real Anthropic only.

    Anthropic-compatible endpoints may not implement Anthropic's
    `web_fetch_20250910` server-side tool. Detect via `ANTHROPIC_BASE_URL` —
    empty/unset means the Anthropic SDK default (real Anthropic).
    """
    base_url = os.environ.get('ANTHROPIC_BASE_URL', '')
    if not base_url or 'api.anthropic.com' in base_url:
        return [NativeTool(WebFetchTool())]
    return []


def _mcp_protocol_error_message(error: Exception) -> str | None:
    """The message of a bare `McpError` that `MCPToolset` left uncaught, else `None`.

    `MCPToolset.direct_call_tool` converts a `ToolError`, a result-level error, and an
    `ExceptionGroup` of tool/protocol errors into a model-visible `ModelRetry`, but a *bare*
    `McpError` — what gh-aw's MCP gateway raises for a validation rejection such as an
    empty-bodied `submit_pull_request_review` — matches none of those. The tool-execute hook
    sees it before the task group upstream re-wraps it into the fatal `ExceptionGroup`, so
    recognising the bare error here is enough to hand it back to the model.
    """
    if isinstance(error, McpError):
        return str(error)
    return None


@dataclass
class _RecoverMCPToolErrors(AbstractCapability[object]):
    """Hand escaped MCP protocol errors back to the model as recoverable `error:` results.

    Without this, a bare `McpError` from an MCP server crashes the whole run (an unhandled
    `ExceptionGroup`), which gh-aw can only recover from by re-invoking the agent wholesale.
    Returning the same `error:` string the shim's function tools use lets the model correct
    the call in-run — e.g. resubmit the review with a non-empty body.
    """

    async def on_tool_execute_error(
        self,
        ctx: RunContext[object],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: object,
        error: Exception,
    ) -> str:
        message = _mcp_protocol_error_message(error)
        if message is None:
            raise error
        logger.warning('MCP tool %r failed — returned to model as a recoverable error: %s', call.tool_name, message)
        return f'error: {message}'


def _mark_timed_out(span: Span) -> None:
    span.set_attribute('error.type', 'timeout')
    span.set_status(Status(StatusCode.ERROR, 'timeout'))


def _wall_clock_cap_expired() -> bool:
    return any(expired() for expired in _wall_clock_caps.get())


@dataclass
class _MarkTimedOutSpans(AbstractCapability[object]):
    """Mark the agent-run and model-request spans that one of the shim's wall-clock caps cuts off.

    OpenTelemetry does not record `asyncio.CancelledError` as an error, so a run or request that
    `task`'s or `_run_with_timeout`'s cap cancels ends its span `UNSET`, like work that finished.
    A cut-off request also has no finish reason, and no usage when the provider reports it only at
    the end of the stream. Other cancellations, such as pydantic-ai stopping a request whose stream
    failed, are left as they are; pydantic-ai already records a cancelled tool span as an error.
    Each hook runs inside the span pydantic-ai opens for it.
    """

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        try:
            return await handler()
        except asyncio.CancelledError:
            if _wall_clock_cap_expired():
                _mark_timed_out(get_current_span())
            raise

    async def wrap_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        try:
            return await handler(request_context)
        except asyncio.CancelledError:
            if _wall_clock_cap_expired():
                _mark_timed_out(get_current_span())
            raise


# pydantic-ai's built-in request_limit default of 50 is too low for the
# deep multi-step workflows here; gh-aw's api-proxy still caps the run.
REQUEST_LIMIT = 400
ATTENTION_REQUEST_LIMIT = 50
SUBAGENT_REQUEST_LIMIT = 150
ATTENTION_WORKFLOW = 'Pydantic AI Attention Triage'


def _is_attention_run() -> bool:
    return os.environ.get('GITHUB_WORKFLOW') == ATTENTION_WORKFLOW


def run_request_limit() -> int:
    """Return the workflow-specific cap on model requests."""
    if _is_attention_run():
        return ATTENTION_REQUEST_LIMIT
    return REQUEST_LIMIT


# Hitting `request_limit` raises before the model can say anything, so a run that
# spent its budget reading files ends with nothing posted. The model cannot see its
# own request count, so it is told once the last tenth of the budget starts. The
# text is fixed rather than a countdown: it sits in the system instructions, and a
# changing value would miss the prompt cache on every remaining request.
REQUEST_BUDGET_NOTICE = (
    '## Request budget nearly spent\n\n'
    'This run is within its last tenth of model requests, and reaching the limit '
    'stops it with nothing posted. Stop investigating. Emit the safe output your '
    'task ends with now, from what you have already found.'
)


def request_budget_notice(ctx: RunContext[object]) -> str | None:
    """Warn the model once its remaining requests fall into the last tenth of the run's limit."""
    request_limit = ctx.usage_limits.request_limit if ctx.usage_limits else None
    if request_limit is None or request_limit - ctx.usage.requests > request_limit // 10:
        return None
    return REQUEST_BUDGET_NOTICE


# `output_type=str` ends the run on any text-only response, even when the model
# narrates its next step without calling a tool. gh-aw then reports the run as
# "produced no safe outputs". The safe-outputs MCP server appends every safe
# output to `GH_AW_SAFE_OUTPUTS`, so an absent or empty file means the task is
# not done. The retry budget is cumulative over the run.
NO_SAFE_OUTPUT_RETRIES = 3


def _safe_output_status() -> tuple[bool, list[int] | None]:
    """Return whether output is pending and any Attention candidate IDs still missing decisions."""
    is_attention = _is_attention_run()
    path = os.environ.get('GH_AW_SAFE_OUTPUTS')
    if not path:
        if is_attention:
            raise ValueError('Attention safe-output sink is not configured')
        return False, None
    sink = pathlib.Path(path)
    sink_empty = not (sink.is_file() and sink.read_text(encoding='utf-8').strip())
    if not is_attention or sink_empty:
        return sink_empty, None

    workspace = pathlib.Path(os.environ.get('GITHUB_WORKSPACE', '.'))
    snapshot_path = workspace / 'attention-candidates.json'
    if not snapshot_path.is_file():
        raise ValueError('Attention candidate snapshot is missing')

    snapshot = TypeAdapter(dict[str, object]).validate_json(snapshot_path.read_text(encoding='utf-8'))
    candidate_rows = TypeAdapter(list[dict[str, object]]).validate_python(snapshot['candidates'])
    candidate_numbers = TypeAdapter(list[Annotated[int, Field(ge=1, strict=True)]]).validate_python(
        [candidate['number'] for candidate in candidate_rows]
    )
    if len(candidate_numbers) > 10 or len(candidate_numbers) != len(set(candidate_numbers)):
        raise ValueError('Attention snapshot contains too many or duplicate candidates')
    candidates = set(candidate_numbers)
    if not candidates:
        return sink_empty, []

    recorded: set[int] = set()
    entry_adapter = TypeAdapter(dict[str, object])
    for line in sink.read_text(encoding='utf-8').splitlines():
        try:
            entry = entry_adapter.validate_json(line)
        except ValidationError:
            continue
        item_number = entry.get('item_number')
        if entry.get('type') == 'record_attention_decision' and isinstance(item_number, str):
            try:
                parsed_number = int(item_number)
            except ValueError:
                continue
            if parsed_number > 0 and str(parsed_number) == item_number:
                recorded.add(parsed_number)
    missing = sorted(candidates - recorded)
    return bool(missing), missing


def safe_output_pending() -> bool:
    """Whether a gh-aw run is missing safe output or Attention decisions."""
    try:
        output_pending, _ = _safe_output_status()
    except (KeyError, OSError, ValueError):
        if _is_attention_run():
            return True
        raise
    return output_pending


def require_safe_output(output: str) -> str:
    """Send the model back to work when it ends the run before emitting any safe output."""
    output_pending, missing = _safe_output_status()
    if missing:
        missing_ids = ', '.join(str(number) for number in missing)
        logger.warning('attention decisions missing for candidate IDs %s; requesting completion', missing_ids)
        raise ModelRetry(
            'Record an attention decision for each missing candidate ID: '
            f'{missing_ids}. Keep the decisions already recorded and call '
            '`record_attention_decision` only for these missing IDs.'
        )
    if not output_pending:
        return output
    logger.warning('run ended with no safe output emitted; sending the model back')
    raise ModelRetry(
        'You ended your turn without emitting a safe output, so nothing has been posted. '
        'Continue the task and finish by calling the safe-output tool it ends with, '
        'or `noop` if there is nothing to report.'
    )


# Per-request HTTP timeout for every LLM call. The read timeout prevents a
# streaming connection that stops sending data from hanging indefinitely. Two
# minutes is generous enough for large generations. SDK-level retries cover
# transient 429/5xx before raising.
_LLM_TIMEOUT = httpx2.Timeout(timeout=120.0, connect=10.0)
_LLM_MAX_RETRIES = 4

# Transient rate limits can outlast the SDK's retry window. Retry 429 responses
# below the SDK with jittered exponential backoff for up to this long. Jitter
# spreads retries when parallel sub-agents hit the limit together.
RATE_LIMIT_RETRY_SECS = 60

# Z.ai 429 codes for account and plan limits, which cannot clear inside the retry
# window. Other 429s, such as `1302` request-rate limits and `1305` overload, are transient.
# Keep these codes aligned with https://docs.z.ai/api-reference/api-code.
_ZAI_PLAN_LIMIT_CODES = frozenset(
    {'1113', '1308', '1309', '1310', '1311', '1313', '1314', '1315', '1316', '1317', '1318', '1319', '1320', '1321'}
)
_ERROR_BODY_ADAPTER = TypeAdapter(dict[str, object])


def _zai_error_code(body: object) -> str | None:
    """Return the `error.code` of a decoded Z.ai error body, or `None` when it has none."""
    try:
        error = _ERROR_BODY_ADAPTER.validate_python(_ERROR_BODY_ADAPTER.validate_python(body).get('error'))
    except ValidationError:
        return None
    code = error.get('code')
    return str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else None


def _give_up_on_rate_limit(state: RetryCallState) -> httpx2.Response | None:
    """Hand the last 429 to the SDK, marked so the SDK does not retry it again.

    Without the mark, each of the SDK's own `_LLM_MAX_RETRIES` would open a fresh
    retry window here, and a persistent 429 would hold one request for minutes.
    Other errors keep the SDK's retries.
    """
    if state.outcome is None:
        return None
    response: httpx2.Response = state.outcome.result()
    response.headers['x-should-retry'] = 'false'
    return response


async def _close_rate_limited_response(state: RetryCallState) -> None:
    """Release a 429 response's connection before the next attempt replaces it."""
    if state.outcome is not None and not state.outcome.failed:
        await state.outcome.result().aclose()


def rate_limit_retry_transport(wrapped: httpx2.AsyncBaseTransport | None = None) -> AsyncHTTPX2TenacityTransport:
    """Transport that retries retryable 429 responses and preserves the final response for the SDK.

    Handing back the response rather than raising keeps the SDK's `RateLimitError`,
    with the provider's error body, as what a run that stays rate-limited fails with.
    """

    async def should_retry(response: httpx2.Response) -> bool:
        if response.status_code != 429:
            return False
        try:
            # `aread()` buffers the body, so the SDK can still consume the
            # original response when this 429 is returned without retrying.
            body = _ERROR_BODY_ADAPTER.validate_json(await response.aread())
        except ValidationError:
            return True
        if _zai_error_code(body) in _ZAI_PLAN_LIMIT_CODES:
            response.headers['x-should-retry'] = 'false'
            return False
        return True

    return AsyncHTTPX2TenacityTransport(
        RetryConfig(
            retry=retry_if_result(should_retry),
            wait=wait_random_exponential(multiplier=1, max=16),
            stop=stop_after_delay(RATE_LIMIT_RETRY_SECS),
            before_sleep=_close_rate_limited_response,
            retry_error_callback=_give_up_on_rate_limit,
        ),
        wrapped=wrapped,
    )


# Wall-clock caps (seconds).  These are last-resort guards on top of the
# per-request timeout so a burst of slow requests can't accumulate forever.
#
# The run cap must land just under the *job's* `timeout-minutes` so the agent
# stops itself and emits a result, rather than being killed mid-flight by
# Actions with nothing to show.  Hardcoding 28 min silently ignored any workflow
# that raised its own `timeout-minutes`, which is how `roundtrip-sweep` spent
# three weeks timing out one minute under a limit it had already outgrown
# (#6766 F6).
#
# gh-aw's own `GH_AW_TIMEOUT_MINUTES` is only set on the "Handle agent failure"
# step, so it never reaches this process.  A workflow therefore declares its job
# timeout as `PYDANTIC_AI_JOB_TIMEOUT_MINUTES` in its top-level `env:`, which
# does propagate into the agent container.  `agentic_workflow_guard.py` enforces
# that the two stay equal, so the duplication cannot drift.
DEFAULT_JOB_TIMEOUT_MINS = 30
# Headroom for container teardown and artifact upload after the agent stops.
JOB_TIMEOUT_HEADROOM_MINS = 2


def _run_timeout_secs() -> int:
    """Wall-clock budget for one agent run, derived from the job's own timeout."""
    raw = os.environ.get('PYDANTIC_AI_JOB_TIMEOUT_MINUTES') or os.environ.get('GH_AW_TIMEOUT_MINUTES') or ''
    try:
        job_mins = int(raw)
    except ValueError:
        job_mins = DEFAULT_JOB_TIMEOUT_MINS
    if job_mins <= JOB_TIMEOUT_HEADROOM_MINS:
        job_mins = DEFAULT_JOB_TIMEOUT_MINS
    return (job_mins - JOB_TIMEOUT_HEADROOM_MINS) * 60


SUBAGENT_TIMEOUT_SECS = 15 * 60  # 15 min per Task sub-agent
COMPACTION_TIMEOUT_SECS = 120  # 2 min for the compaction summariser call

# The run's last minutes belong to the parent: sub-agents are cut off before they start,
# and the parent is then told to emit its safe output. Successful `CI Review` runs need
# up to about 4 min between their last sub-agent returning and their posted review.
RUN_TIME_RESERVE_SECS = 5 * 60
TIME_BUDGET_NOTICE = (
    '## Time budget nearly spent\n\n'
    f'This run has under {RUN_TIME_RESERVE_SECS // 60} minutes of wall-clock time left, and reaching '
    'the limit stops it with nothing posted. Stop investigating. Emit the safe output your '
    'task ends with now, from what you have already found.'
)

# Monotonic time at which `_run_with_timeout` cancels the run; `None` outside a run.
_run_deadline: float | None = None

# Whether each wall-clock cap the current work runs under has expired, innermost last.
_wall_clock_caps: ContextVar[tuple[Callable[[], bool], ...]] = ContextVar('_wall_clock_caps', default=())


@asynccontextmanager
async def _wall_clock_cap(seconds: float) -> AsyncGenerator[None]:
    """`asyncio.timeout` that marks the span holding it when it expires.

    It is registered so `_MarkTimedOutSpans` can tell the cancellations it causes from others, and a
    `TimeoutError` raised by the work it caps leaves it unexpired. `task`'s cap marks the
    `execute_tool Task` span, which otherwise looks like a successful call.
    """
    cap = asyncio.timeout(seconds)
    token = _wall_clock_caps.set((*_wall_clock_caps.get(), cap.expired))
    try:
        async with cap:
            yield
    except TimeoutError:
        if cap.expired():
            _mark_timed_out(get_current_span())
        raise
    finally:
        _wall_clock_caps.reset(token)


def _run_seconds_left() -> float | None:
    """Wall-clock seconds before the run is cancelled, or `None` when no run budget applies."""
    return None if _run_deadline is None else _run_deadline - time.monotonic()


def time_budget_notice() -> str | None:
    """Warn the model once the run's remaining wall-clock time falls inside the parent's reserve."""
    seconds_left = _run_seconds_left()
    if seconds_left is None or seconds_left > RUN_TIME_RESERVE_SECS:
        return None
    return TIME_BUDGET_NOTICE


# gh-aw's `safeoutputs` MCP server, under the shim's Claude Code tool-name prefix.
SAFE_OUTPUT_TOOL_PREFIX = 'mcp__safeoutputs__'


def hold_time_reserve_for_safe_output(
    _ctx: RunContext[object], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    """Offer only the safe-output tools once the run is inside the parent's time reserve.

    The notice alone does not stop a model from investigating into the reserve. The request in
    flight when the reserve starts still completes and its tool calls still run; the notice
    stays, so the model knows why its other tools are gone.
    """
    if time_budget_notice() is None:
        return tool_defs
    return [tool_def for tool_def in tool_defs if tool_def.name.startswith(SAFE_OUTPUT_TOOL_PREFIX)]


class RunTimeLimitExceeded(Exception):
    """The run used its whole wall-clock budget, which is not a provider timeout."""


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
    '## GitHub issue and PR search\n\n'
    'Use the context prefetched for this workflow instead of enumerating GitHub '
    'through the proxied gh CLI; list/search requests from inside the sandbox '
    'are blocked or can stall until the workflow times out. Issue-filing sweeps '
    'provide `/tmp/gh-aw/agent/github-context/open-issues.json` and '
    '`open-pull-requests.json`; filter their local JSON with jq. PR reviewers '
    'use `$GITHUB_WORKSPACE/.review-context/`; the stale-issues workflow uses '
    '`/tmp/gh-aw/agent/open-issues.tsv` and `/tmp/gh-aw/agent/issues/`. Do NOT '
    'run `gh issue list`, `gh pr list`, `gh search`, or a paginated/list `gh api` '
    'request from inside the agent. Narrow per-item reads may still be used '
    'after the local corpus identifies a specific item. If required prefetched '
    'context is missing or unreadable, call `mcp__safeoutputs__noop` and report '
    'the missing data instead of attempting a list request through gh-proxy.'
)

# The real task spec rides in `instructions=`; the user message is a trigger.
RUN_TRIGGER = 'Begin the task per the instructions above.'

SUBAGENT_INSTRUCTIONS = (
    'You are a focused, read-only sub-agent. You can read files, search the '
    'codebase, and fetch web content, but you cannot modify the workspace or '
    'shell out. Investigate the task you were given and return a concise, '
    'evidence-grounded answer to your caller — do not try to act on it.'
)

# A sub-agent that hits its timeout has often gathered most of what it needs. `task` then makes
# one request without tools on its saved history to recover its answer.
SUBAGENT_RESUME_PROMPT = (
    'Your time is up and your tools are no longer available. Answer your caller now, in the '
    'format your task asks for, from what you have already found. Name everything you were '
    'asked to cover that you did not finish.'
)

# History compaction (pydantic-ai `ProcessHistory` capability). Two stages
# inside one callback: a cheap dedup+truncate trim, then an LLM summary as
# fallback. `Agent(instructions=...)` is re-applied on every request, so
# the workflow prompt is never in the message list and never compacted.

# ~100k tokens at 4 chars/token = half of a 200k window.
COMPACTION_TRIGGER_CHARS = 400_000
COMPACTION_KEEP_RECENT = 10
TOOL_RESULT_HEAD_TAIL_CHARS = 4_000
TOOL_RESULT_TRIM_THRESHOLD = 10_000
COMPACTION_TRANSCRIPT_MAX_CHARS = 80_000

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
    """Best-effort text rendering of any pydantic-ai message part."""
    if isinstance(part, (ToolCallPart, NativeToolCallPart, ToolSearchCallPart, NativeToolSearchCallPart)):
        return f'{part.tool_name}({part.args_as_dict()!r})'
    if isinstance(part, ToolAvailabilityDeltaPart):
        return f'tool availability changed: tools_added={part.tools_added!r}'
    if isinstance(part, InstructionBaselinePart):
        # Records what the request's instructions were made of; the transcript leaves instructions out.
        return ''
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
    """Char-count proxy for token cost — used as the compaction trigger."""
    return sum(len(_part_text(part)) for m in messages for part in m.parts)


def _head_tail(text: str, side: int) -> str:
    """Keep the first and last `side` chars, mark the elided middle."""
    skipped = len(text) - side * 2
    return f'{text[:side]}\n…[trimmed {skipped} chars]…\n{text[-side:]}'


def _superseded_read_calls(messages: list[ModelMessage]) -> tuple[set[str], dict[str, str]]:
    """For each `Read` call, key on (path, offset, limit); older calls with the same key are superseded."""
    label_by_call_id: dict[str, str] = {}
    latest_for_args: dict[tuple[str, object, object], str] = {}
    superseded: set[str] = set()
    for m in messages:
        for p in m.parts:
            if not (isinstance(p, ToolCallPart) and p.tool_name == 'Read'):
                continue
            args = p.args_as_dict()
            if not isinstance(args, dict):
                continue
            path = args.get('file_path')
            if not isinstance(path, str):
                continue
            offset, limit = args.get('offset'), args.get('limit')
            label = path if offset is None and limit is None else f'{path}[offset={offset!r}, limit={limit!r}]'
            label_by_call_id[p.tool_call_id] = label
            key = (path, offset, limit)
            prior = latest_for_args.get(key)
            if prior is not None:
                superseded.add(prior)
            latest_for_args[key] = p.tool_call_id
    return superseded, label_by_call_id


def _trim_tool_results(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Dedupe re-reads of the same file slice and head/tail-truncate oversized older tool returns."""
    if len(messages) <= COMPACTION_KEEP_RECENT:
        return messages
    superseded, label_by_call_id = _superseded_read_calls(messages)

    tail_start = len(messages) - COMPACTION_KEEP_RECENT
    out: list[ModelMessage] = []
    dedup_count = truncate_count = bytes_saved = 0

    def _rewrite(part: ModelRequestPart | ModelResponsePart) -> ModelRequestPart | ModelResponsePart:
        nonlocal dedup_count, truncate_count, bytes_saved
        if not isinstance(part, ToolReturnPart):
            return part
        if part.tool_call_id in superseded:
            new_content = f'[superseded read: {label_by_call_id[part.tool_call_id]} — see later read with same args]'
            bytes_saved += len(str(part.content)) - len(new_content)
            dedup_count += 1
            return dataclasses.replace(part, content=new_content)
        content = str(part.content)
        if len(content) > TOOL_RESULT_TRIM_THRESHOLD:
            new_content = _head_tail(content, TOOL_RESULT_HEAD_TAIL_CHARS)
            bytes_saved += len(content) - len(new_content)
            truncate_count += 1
            return dataclasses.replace(part, content=new_content)
        return part

    for idx, m in enumerate(messages):
        if idx >= tail_start:
            out.append(m)
            continue
        new_parts = [_rewrite(p) for p in m.parts]
        out.append(dataclasses.replace(m, parts=new_parts) if new_parts != list(m.parts) else m)

    if dedup_count or truncate_count:
        logger.info(
            'trim: deduped %d superseded read(s), truncated %d oversized result(s), saved %d chars',
            dedup_count,
            truncate_count,
            bytes_saved,
        )
        emit(
            {
                'type': 'system',
                'subtype': 'compaction_trim',
                'deduped_reads': dedup_count,
                'truncated_results': truncate_count,
                'chars_saved': bytes_saved,
            }
        )
        return out
    return messages


_SYNTHETIC_SUMMARY_TAG = '[compacted history]'


def _is_synthetic_summary(message: ModelMessage) -> bool:
    """A `ModelRequest` we synthesised in a prior `_compact_history` round."""
    if not isinstance(message, ModelRequest):
        return False
    parts = message.parts
    return (
        len(parts) == 1
        and isinstance(parts[0], UserPromptPart)
        and str(parts[0].content).startswith(_SYNTHETIC_SUMMARY_TAG)
    )


async def _compact_history(ctx: RunContext[object], messages: list[ModelMessage]) -> list[ModelMessage]:
    """Cheap trim first; LLM-summarise the middle as fallback if still over budget."""
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
        'compaction summary firing: %d chars / %d messages -> summarising %d middle, keeping last %d',
        size,
        len(trimmed),
        len(middle),
        COMPACTION_KEEP_RECENT,
    )
    emit(
        {
            'type': 'system',
            'subtype': 'compaction_summary_start',
            'history_chars': size,
            'history_messages': len(trimmed),
            'middle_messages': len(middle),
            'keep_recent': COMPACTION_KEEP_RECENT,
        }
    )
    # Preserve any earlier-round synthetic at the head of the middle so a
    # fallback (`return [prior_synthetic, *tail]`) doesn't silently forget
    # the entire run's compacted history.
    prior_synthetic = middle[0] if middle and _is_synthetic_summary(middle[0]) else None

    # Fresh `RunUsage` so `request_limit=2` bounds the summariser, not
    # (parent + summariser). Merge the totals back regardless of outcome.
    sub_usage = RunUsage()
    # `RunContext.model` is the wider `AbstractModel`; the shim only ever runs regular agents, so it is
    # always a `Model` here.
    model = ctx.model
    assert isinstance(model, Model)
    try:
        r = await asyncio.wait_for(
            Agent(cast('Model[Any]', model), instructions=COMPACTION_SUMMARY_INSTRUCTIONS).run(
                f'Transcript to summarise:\n\n{transcript[:COMPACTION_TRANSCRIPT_MAX_CHARS]}',
                usage_limits=UsageLimits(request_limit=2),
                usage=sub_usage,
            ),
            timeout=COMPACTION_TIMEOUT_SECS,
        )
        summary = str(r.output or '').strip() or '(empty summary)'
    except Exception as exc:
        # Well-handled fallback: the run continues on the trimmed history, so the
        # stack is kept available (`exc_info`) without escalating the log level.
        # A bare `TimeoutError` stringifies to '' — fall back to the type name so
        # the emitted `error` is never empty.
        ctx.usage.incr(sub_usage)
        detail = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
        logger.warning('compaction summarisation failed (%s); falling back', detail, exc_info=True)
        emit({'type': 'system', 'subtype': 'compaction_summary_failed', 'error': detail})
        return [prior_synthetic, *tail] if prior_synthetic else tail
    ctx.usage.incr(sub_usage)
    # If the summariser produces output larger than the middle it's replacing,
    # the next compaction round would trip on the same too-large synthetic
    # and never converge — fall back to the prior synthetic + tail.
    middle_size = _history_size_chars(middle)
    if len(summary) >= middle_size:
        logger.info('compaction summary discarded (%d >= %d chars); falling back', len(summary), middle_size)
        emit(
            {
                'type': 'system',
                'subtype': 'compaction_summary_discarded',
                'summary_chars': len(summary),
                'middle_chars': middle_size,
            }
        )
        return [prior_synthetic, *tail] if prior_synthetic else tail
    logger.info(
        'compaction summary done: %d middle messages (%d chars) -> %d-char summary',
        len(middle),
        middle_size,
        len(summary),
    )
    emit(
        {
            'type': 'system',
            'subtype': 'compaction_summary_done',
            'middle_messages': len(middle),
            'middle_chars': middle_size,
            'summary_chars': len(summary),
            'input_tokens': sub_usage.input_tokens,
            'output_tokens': sub_usage.output_tokens,
        }
    )
    synthetic = ModelRequest(parts=[UserPromptPart(content=f'{_SYNTHETIC_SUMMARY_TAG}\n{summary}')])
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
    under gh-aw. The explicit `AsyncAnthropic(auth_token=...)` path lets
    the SDK construct a request with a bearer `Authorization` header
    (the non-secret placeholder under AWF). The api-proxy sidecar strips
    client-supplied `Authorization` and `x-api-key` headers, then injects
    the real `x-api-key` from its isolated `ANTHROPIC_API_KEY`.
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
        http_client=httpx2.AsyncClient(transport=rate_limit_retry_transport(), timeout=_LLM_TIMEOUT),
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
        # Also instruments `httpx2`, the model client's library, by patching its
        # `AsyncHTTPTransport`: every attempt `rate_limit_retry_transport` makes gets its own
        # span. Request bodies would repeat each `chat` span's `gen_ai.input.messages`.
        logfire.instrument_httpx(capture_headers=True, capture_response_body=True)
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
    """Apply allow-list and read-only filters to the Claude Code toolset."""
    plan = permission_mode == 'plan'
    attention_read_only = _is_attention_run()

    def predicate(_ctx: RunContext[object], tool_def: ToolDefinition) -> bool:
        name = tool_def.name
        if allowed is not None and name not in allowed:
            return False
        if (plan or attention_read_only) and name in MUTATING_TOOLS:
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
    num_turns: int = 0,
    duration_ms: int = 0,
    error: BaseException | None = None,
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
    run_attempt: int | None = None
    raw_run_attempt = os.environ.get('PYDANTIC_AI_RUN_ATTEMPT')
    if raw_run_attempt is not None:
        try:
            parsed_run_attempt = int(raw_run_attempt)
        except ValueError:
            pass
        else:
            if parsed_run_attempt > 0:
                run_attempt = parsed_run_attempt

    provider_health: dict[str, object] = {
        'workflow': os.environ.get('GITHUB_WORKFLOW'),
        'task_key': os.environ.get('PYDANTIC_AI_TASK_KEY'),
        'trigger_event': os.environ.get('PYDANTIC_AI_TRIGGER_EVENT'),
        'run_attempt': run_attempt,
    }
    if is_error:
        provider_health['failure'] = _failure_details(error)
    result: dict[str, object] = {
        'type': 'result',
        'subtype': 'error' if is_error else 'success',
        'is_error': is_error,
        'result': text,
        'session_id': session_id,
        'num_turns': num_turns,
        'duration_ms': duration_ms,
        'total_cost_usd': 0,
        'usage': token_usage,
        'provider_health': provider_health,
    }
    emit(result)


def _failure_details(error: BaseException | None) -> dict[str, object]:
    """Return a safe, machine-readable classification while preserving the original error text."""
    kind = 'other'
    http_status: int | None = None
    if isinstance(error, UsageLimitExceeded):
        kind = 'request_limit'
    elif isinstance(error, RunTimeLimitExceeded):
        kind = 'time_limit'
    elif isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        kind = 'timeout'
    elif isinstance(error, ModelAPIError) and isinstance(error.__cause__, (APITimeoutError, httpx2.TimeoutException)):
        kind = 'timeout'
    elif isinstance(error, ModelHTTPError):
        http_status = error.status_code
        if error.status_code in (401, 403):
            kind = 'authentication'
        elif error.status_code == 429:
            kind = 'plan_limit' if _zai_error_code(error.body) in _ZAI_PLAN_LIMIT_CODES else 'rate_limit'
    failure: dict[str, object] = {'kind': kind}
    if http_status is not None:
        failure['http_status'] = http_status
    return failure


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
            # `outcome='retried'` means the call didn't work and pydantic-ai is asking the model to
            # make it again. Tag it so gh-aw doesn't read it as success.
            is_retry = event.part.outcome == 'retried'
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


# Requests granted to sub-agents that have not returned yet. Their usage reaches the
# parent's `ctx.usage` only on return, so parallel `Task` calls would otherwise each
# see the same headroom and jointly overshoot it.
_subagent_requests_in_flight: int = 0


def _subagent_request_limit(ctx: RunContext[object]) -> int:
    """Requests a new sub-agent may spend without eating into the parent's final tenth.

    The parent's budget notice only fires if the parent itself still has requests left
    when that tenth starts; a sub-agent that returns past it would skip the notice.
    """
    request_limit = ctx.usage_limits.request_limit if ctx.usage_limits else None
    if request_limit is None:
        return SUBAGENT_REQUEST_LIMIT
    headroom = request_limit - request_limit // 10 - ctx.usage.requests - _subagent_requests_in_flight
    return min(SUBAGENT_REQUEST_LIMIT, headroom)


def _subagent_time_limit() -> float:
    """Seconds a sub-agent may run from now without eating into the parent's time reserve."""
    seconds_left = _run_seconds_left()
    if seconds_left is None:
        return SUBAGENT_TIMEOUT_SECS
    return min(SUBAGENT_TIMEOUT_SECS, seconds_left - RUN_TIME_RESERVE_SECS)


async def task(ctx: RunContext[object], description: str, prompt: str) -> str:
    """Claude's `Task` tool: spawn a read-only sub-agent on `ctx.model`."""
    global _subagent_requests_in_flight
    sub_request_limit = _subagent_request_limit(ctx)
    if sub_request_limit <= 0:
        logger.info('Task refused, request budget nearly spent: %s', description[:120])
        return 'error: the request budget is nearly spent; finish the task from what you already have'
    timeout = _subagent_time_limit()
    if timeout <= 0:
        logger.info('Task refused, time budget nearly spent: %s', description[:120])
        return 'error: the time budget is nearly spent; finish the task from what you already have'
    logger.info('Task spawn: %s', description[:120])
    # Fresh dedupe set per sub-agent — otherwise inheriting the parent's
    # `seen` AGENTS.md set would silently hide context the sub-agent needs.
    reset_context_state()
    sub_toolset = select_claude_code_toolset(READ_ONLY_SUBAGENT_TOOLS, permission_mode=None, task=None)
    # `RunContext.model` is the wider `AbstractModel`; the shim only ever runs regular agents, so it is
    # always a `Model` here.
    model = ctx.model
    assert isinstance(model, Model)
    instructions = [INSTRUCTIONS, SUBAGENT_INSTRUCTIONS, prompt]
    sub = Agent(
        cast('Model[Any]', model),
        instructions=instructions,
        toolsets=[sub_toolset],
        capabilities=[
            local_workspace(),
            _MarkTimedOutSpans(),
            *_anthropic_native_capabilities(),
            ProcessEventStream(_stream_events),
        ],
    )
    # Fresh `RunUsage` so `sub_request_limit` bounds the sub-agent, not
    # (parent + sub). Merge the deltas back regardless of success/failure.
    sub_usage = RunUsage()
    sub_messages: list[ModelMessage] = []
    _subagent_requests_in_flight += sub_request_limit
    try:
        with capture_run_messages() as sub_messages:
            async with _wall_clock_cap(timeout):
                result = await sub.run(
                    RUN_TRIGGER, usage_limits=UsageLimits(request_limit=sub_request_limit), usage=sub_usage
                )
    except TimeoutError:
        # A bare `TimeoutError` stringifies to '' — without an explicit message
        # the model (and the log) would see `sub-agent failed:` with no payload.
        ctx.usage.incr(sub_usage)
        logger.error('sub-agent timed out after %.0f min: %s', timeout / 60, description[:120])
        timeout_error = f'error: sub-agent timed out after {math.ceil(timeout / 60)}min'
        # The resume is the sub-agent's last request, so it spends from the sub-agent's own request grant,
        # which stays in flight until `finally`. A response the cut left partial is dropped, and
        # `repair_messages` closes out tool calls the cut left unanswered.
        history = [m for m in sub_messages if not (isinstance(m, ModelResponse) and m.state == 'interrupted')]
        resume_timeout = _subagent_time_limit()
        if (
            resume_timeout <= 0
            or sub_usage.requests >= sub_request_limit
            or not any(isinstance(m, ModelResponse) for m in history)
        ):
            return timeout_error
        resume_usage = RunUsage()
        try:
            async with _wall_clock_cap(resume_timeout):
                # Streamed like the sub-agent, so a long report is not cut by the client's read timeout.
                resumed = await Agent(
                    cast('Model[Any]', model),
                    name='delegate-resume',
                    instructions=instructions,
                    capabilities=[_MarkTimedOutSpans(), ProcessEventStream(_stream_events)],
                ).run(
                    SUBAGENT_RESUME_PROMPT,
                    message_history=repair_messages(history),
                    usage_limits=UsageLimits(request_limit=1),
                    usage=resume_usage,
                )
        except Exception:
            logger.exception('timed-out sub-agent could not be resumed: %s', description[:120])
            return timeout_error
        finally:
            ctx.usage.incr(resume_usage)
        logger.info('Task resumed after its timeout: %s', description[:120])
        return (
            f'note: the sub-agent timed out after {math.ceil(timeout / 60)}min and wrote this report afterwards '
            f'from what it had already found, without further tool calls.\n\n{resumed.output}'
        )
    except Exception as exc:
        # The parent agent reacts to the returned string, but a sub-agent can hit
        # the same nested `ExceptionGroup`/`McpError` as the main run — log the
        # full stack so the failure isn't reduced to a one-line repr in the logs.
        ctx.usage.incr(sub_usage)
        logger.exception('sub-agent failed: %s', description[:120])
        return f'error: sub-agent failed: {exc}'
    finally:
        _subagent_requests_in_flight -= sub_request_limit
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
    global _run_deadline
    budget = _run_timeout_secs()
    _run_deadline = time.monotonic() + budget
    usage = RunUsage()
    try:
        async with AsyncExitStack() as stack:
            if isinstance(model, AnthropicModel):
                await stack.enter_async_context(model.client)
            async with _wall_clock_cap(budget):
                return await run(prompt, model, label, claude_code_toolset, mcp_servers, session_id, usage=usage)
    except TimeoutError:
        logger.error('run timed out after %.0f min', budget / 60)
        emit_result(
            f'run timed out after {budget // 60}min',
            usage=usage,
            session_id=session_id,
            is_error=True,
            num_turns=usage.requests,
            error=RunTimeLimitExceeded(),
        )
        return 1
    except Exception as exc:
        logger.exception('runner failed outside agent execution')
        emit_result(
            f'agent run failed: {exc}',
            usage=usage,
            session_id=session_id,
            is_error=True,
            num_turns=usage.requests,
            error=exc,
        )
        return 1
    finally:
        _run_deadline = None


async def run(
    prompt: str,
    model: Model,
    label: str,
    claude_code_toolset: AbstractToolset[object],
    mcp_servers: list[AbstractToolset[object]],
    session_id: str,
    usage: RunUsage | None = None,
) -> int:
    """Run one agent turn and emit Claude-shape stream-json. Always emits a `result` line."""
    reset_context_state()
    run_usage = usage or RunUsage()
    agent: Agent[object, str] = Agent(
        model,
        instructions=[INSTRUCTIONS, prompt, request_budget_notice, time_budget_notice],
        retries={'output': NO_SAFE_OUTPUT_RETRIES},
        toolsets=[claude_code_toolset, *mcp_servers],
        capabilities=[
            # The Claude tools act on `ctx.workspace`: the checkout at `$GITHUB_WORKSPACE`.
            local_workspace(),
            _RecoverMCPToolErrors(),
            *_anthropic_native_capabilities(),
            ProcessHistory(_compact_history),
            ProcessEventStream(_stream_events),
            _MarkTimedOutSpans(),
            PrepareTools(hold_time_reserve_for_safe_output),
        ],
    )
    agent.output_validator(require_safe_output)
    limits = UsageLimits(request_limit=run_request_limit())
    emit({'type': 'system', 'subtype': 'init', 'session_id': session_id, 'model': label})

    started = time.perf_counter()
    try:
        async with agent:
            result = await agent.run(RUN_TRIGGER, usage_limits=limits, usage=run_usage)
    except Exception as exc:
        # The limit is checked before a request, so a safe-output tool called on the
        # last one has already run: the task is done, only the closing turn is lost.
        if isinstance(exc, UsageLimitExceeded) and not safe_output_pending():
            logger.warning('request limit reached after the safe output was emitted: %s', exc)
            emit_result(
                'request limit reached after the safe output was emitted',
                usage=run_usage,
                session_id=session_id,
                num_turns=run_usage.requests,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
            log_safe_outputs_state()
            return 0
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
            usage=run_usage,
            session_id=session_id,
            is_error=True,
            num_turns=run_usage.requests,
            duration_ms=round((time.perf_counter() - started) * 1000),
            error=exc,
        )
        return 1

    duration_ms = round((time.perf_counter() - started) * 1000)
    messages = result.all_messages()
    tool_calls = count_tool_calls(messages)
    num_turns = run_usage.requests
    logger.info('tool calls observed: %d, turns: %d', tool_calls, num_turns)

    text = str(result.output or '')
    emit({'type': 'assistant', 'message': {'role': 'assistant', 'content': text}})

    emit_result(text, run_usage, session_id, num_turns=num_turns, duration_ms=duration_ms)
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
        task_tool = None if _is_attention_run() else task
        claude_code_toolset = select_claude_code_toolset(args.allowed_tools, args.permission_mode, task=task_tool)
        mcp_servers = build_mcp_servers(args)
        logger.info(
            'model=%s permission_mode=%s request_limit=%d claude_code_tool_names=%s mcp_servers=%d prompt_chars=%d',
            label,
            args.permission_mode or '(none)',
            run_request_limit(),
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
        emit_result(f'shim startup failed: {exc}', usage=None, session_id=session_id, is_error=True, error=exc)
        return 1
    except Exception as exc:
        # A real crash before the agent run (model build, MCP load, …) — dump the
        # full stack so a blind FATAL doesn't cost another long investigation.
        logger.exception('FATAL startup error')
        emit_result(f'shim startup failed: {exc}', usage=None, session_id=session_id, is_error=True, error=exc)
        return 1
