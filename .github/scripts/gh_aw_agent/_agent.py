"""The composition loaded by the public `pydantic-ai` engine."""

from __future__ import annotations

import os
from fnmatch import fnmatchcase

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace, ProcessHistory
from pydantic_ai.messages import ModelMessage
from pydantic_ai_harness import Coder
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

from ._budgets import (
    DelegateReserve,
    LimitAfterOutputIsSuccess,
    RecordRunTotals,
    RequestBudget,
    WallClockBudget,
    request_budget_notice,
    run_timeout_secs,
    time_budget_notice,
)
from ._compaction import compact_history, require_safe_output
from ._config import is_review_run
from ._policies import RecoverMCPToolErrors, ToolPolicy
from ._rate_limit import AnthropicRetryTransport
from ._result import RecordResult
from ._telemetry import configure_observability

INSTRUCTIONS = (
    '## Parallel tool calls\n\n'
    'The model supports parallel tool calls. When multiple reads, searches, or '
    "lookups are independent — meaning one doesn't need another's result — "
    'issue them all in the same response. They execute concurrently. Only '
    'chain sequentially when one call genuinely needs a previous result.\n\n'
    '## File reading\n\n'
    'Read files in large ranges (500+ lines per call). Most Python source files '
    'fit in one or two calls. Avoid reading 30-80 lines at a time. Use '
    '`list_files` with a glob and `grep` for search.\n\n'
    '## Dev environment\n\n'
    'The repo is checked out at the workspace root. Dev dependencies are NOT '
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
    '/tmp/gh-aw/agent/open-issues.tsv and /tmp/gh-aw/agent/issues/. Do NOT '
    'run `gh issue list`, `gh pr list`, `gh search`, or a paginated/list `gh api` '
    'request from inside the agent. Narrow per-item reads may still be used '
    'after the local corpus identifies a specific item. If required prefetched '
    'context is missing or unreadable, report the missing data instead of attempting '
    'a list request through gh-proxy. CI Review and UI Security must submit a COMMENT '
    'review explaining incomplete coverage; other workflows may use `safeoutputs_noop`.'
)

RUN_TRIGGER = 'Begin the task per the instructions above.'


async def _process_history(ctx: RunContext[object], messages: list[ModelMessage]) -> list[ModelMessage]:
    return await compact_history(ctx, messages, preserve_prompt=True)


def _child_environment() -> dict[str, str]:
    """The step environment minus provider credentials, which Coder's shell has always withheld."""
    return {
        name: value
        for name, value in os.environ.items()
        if not any(fnmatchcase(name, pattern) for pattern in LLM_API_KEY_ENV_PATTERNS)
    }


def build_agent() -> Agent[object, str]:
    configure_observability()
    instructions = [INSTRUCTIONS, request_budget_notice, time_budget_notice]
    if is_review_run():
        instructions.append(
            'Submit the formal pull-request review before you finish. A noop, inline comment, '
            'or missing-data signal does not finish a review.'
        )
    agent: Agent[object, str] = Agent(
        name='gh-aw-agent',
        instructions=instructions,
        retries={'output': 3},
        capabilities=[
            RecordResult(),
            LocalWorkspace('.', env=_child_environment()),
            Coder(),
            RecoverMCPToolErrors(),
            ToolPolicy(),
            WallClockBudget(run_timeout_secs()),
            AnthropicRetryTransport(),
            RecordRunTotals(),
            RequestBudget(),
            LimitAfterOutputIsSuccess(),
            DelegateReserve(),
            ProcessHistory(_process_history),
        ],
    )
    agent.output_validator(require_safe_output)
    return agent


agent = build_agent()
