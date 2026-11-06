"""Environment-derived workflow identity and budgets."""

from __future__ import annotations

import os

ATTENTION_WORKFLOW = 'Pydantic AI Attention Triage'
CI_REVIEW_WORKFLOW = 'CI Review'
UI_SECURITY_WORKFLOW = 'Pydantic AI UI Security Review'
REVIEW_WORKFLOWS = frozenset({CI_REVIEW_WORKFLOW, UI_SECURITY_WORKFLOW})

DEFAULT_JOB_TIMEOUT_MINS = 30
DEFAULT_REQUEST_LIMIT = 400
ATTENTION_REQUEST_LIMIT = 50
JOB_TIMEOUT_HEADROOM_MINS = 2
RUN_TIME_RESERVE_SECS = 5 * 60
COMPACTION_TIMEOUT_SECS = 120


def workflow_name() -> str:
    return os.environ.get('GITHUB_WORKFLOW', '')


def is_attention_run() -> bool:
    return workflow_name() == ATTENTION_WORKFLOW


def is_review_run() -> bool:
    return workflow_name() in REVIEW_WORKFLOWS


def is_ci_review_run() -> bool:
    return workflow_name() == CI_REVIEW_WORKFLOW


def request_limit() -> int:
    """Model requests one run may spend; gh-aw's AWF cap is 500."""
    if os.environ.get('GITHUB_WORKFLOW') == ATTENTION_WORKFLOW:
        return ATTENTION_REQUEST_LIMIT
    raw = os.environ.get('PYDANTIC_AI_REQUEST_LIMIT')
    try:
        limit = int(raw) if raw else DEFAULT_REQUEST_LIMIT
    except ValueError:
        return DEFAULT_REQUEST_LIMIT
    return limit if limit > 0 else DEFAULT_REQUEST_LIMIT


def shell_disabled() -> bool:
    """Whether the workflow's `tools.bash: []` forbids shell tool exposure."""
    return os.environ.get('PYDANTIC_AI_SHELL_DISABLED', '') == '1'


def run_timeout_secs() -> int:
    """Wall-clock budget for one agent run, derived from the job's own timeout.

    The engine's step is bounded by the action timeout, which kills the process
    without a result. Stopping two minutes early lets the recorder close the
    session and the safe-output ingestion run.
    """
    raw = os.environ.get('PYDANTIC_AI_JOB_TIMEOUT_MINUTES') or ''
    try:
        job_mins = int(raw)
    except ValueError:
        job_mins = DEFAULT_JOB_TIMEOUT_MINS
    if job_mins <= JOB_TIMEOUT_HEADROOM_MINS:
        job_mins = DEFAULT_JOB_TIMEOUT_MINS
    return (job_mins - JOB_TIMEOUT_HEADROOM_MINS) * 60
