"""Terminal provider-health metadata for the workflow monitor.

On main the shim printed a `type: result` line carrying `provider_health` to
stdout and `agent_provider_health.py` read it from `agent-stdio.log`. The
engine sends stdout to devnull, so this module writes the same record once, at
finish, to a file the `agent` artifact already carries
(`/tmp/gh-aw/agent/provider-health-result.json`). The record is unsigned, as it
was on main; signing is a separate decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

from pydantic_ai import AgentRunResult, RunContext, UsageLimitExceeded
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

logger = logging.getLogger(__name__)

RESULT_PATH_ENV = 'PYDANTIC_AI_AGENT_RESULT_PATH'
DEFAULT_RESULT_PATH = '/tmp/gh-aw/agent/provider-health-result.json'

# Z.ai 429 codes for account and plan limits, which cannot clear inside a retry window.
ZAI_PLAN_LIMIT_CODES = frozenset(
    {'1113', '1308', '1309', '1310', '1311', '1313', '1314', '1315', '1316', '1317', '1318', '1319', '1320', '1321'}
)
ZAI_RATE_LIMIT_CODES = frozenset({'1302', '1305'})


class FailureRecord(TypedDict, total=False):
    kind: str
    http_status: int


class ProviderHealthRecord(TypedDict, total=False):
    workflow: str | None
    task_key: str | None
    trigger_event: str | None
    run_attempt: int
    failure: FailureRecord


_ERROR_BODY_ADAPTER = TypeAdapter(dict[str, object])


def _zai_error_code(body: object) -> str | None:
    try:
        error = _ERROR_BODY_ADAPTER.validate_python(_ERROR_BODY_ADAPTER.validate_python(body).get('error'))
    except ValidationError:
        return None
    code = error.get('code')
    return str(code) if isinstance(code, (str, int)) and not isinstance(code, bool) else None


def _failure_details(error: BaseException) -> FailureRecord:
    kind = 'other'
    http_status: int | None = None
    if isinstance(error, UsageLimitExceeded):
        kind = 'request_limit'
    elif isinstance(error, RunTimeLimitExceeded):
        kind = 'time_limit'
    elif isinstance(error, (asyncio.TimeoutError, TimeoutError)):
        kind = 'timeout'
    elif isinstance(error, ModelHTTPError):
        http_status = error.status_code
        if error.status_code in (401, 403):
            kind = 'authentication'
        elif error.status_code == 429:
            code = _zai_error_code(error.body)
            kind = 'plan_limit' if code in ZAI_PLAN_LIMIT_CODES else 'rate_limit'
    elif isinstance(error, ModelAPIError) and isinstance(error.__cause__, (asyncio.TimeoutError, TimeoutError)):
        kind = 'timeout'
    failure: FailureRecord = {'kind': kind}
    if http_status is not None:
        failure['http_status'] = http_status
    return failure


class RunTimeLimitExceeded(Exception):
    """The run used its whole wall-clock budget, which is not a provider timeout."""


def _run_attempt() -> int:
    raw = os.environ.get('PYDANTIC_AI_RUN_ATTEMPT')
    try:
        attempt = int(raw or '')
    except ValueError:
        return 1
    return attempt if attempt > 0 else 1


def _record(error: BaseException | None) -> dict[str, object]:
    provider_health: ProviderHealthRecord = {
        'workflow': os.environ.get('GITHUB_WORKFLOW'),
        'task_key': os.environ.get('PYDANTIC_AI_TASK_KEY'),
        'trigger_event': os.environ.get('PYDANTIC_AI_TRIGGER_EVENT'),
        'run_attempt': _run_attempt(),
    }
    if error is not None:
        provider_health['failure'] = _failure_details(error)
    return {'type': 'result', 'provider_health': provider_health}


def write_result(error: BaseException | None = None) -> None:
    """Write the terminal record where the `agent` artifact already collects files."""
    path = pathlib.Path(os.environ.get(RESULT_PATH_ENV) or DEFAULT_RESULT_PATH)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_record(error)), encoding='utf-8')
    except OSError as exc:
        logger.warning('unable to write provider-health result to %s: %r', path, exc)


@dataclass
class RecordResult(AbstractCapability[object]):
    """Write the run's terminal provider-health record exactly once."""

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        del ctx
        error: BaseException | None = None
        try:
            return await handler()
        except BaseException as exc:
            error = exc
            raise
        finally:
            write_result(error)
