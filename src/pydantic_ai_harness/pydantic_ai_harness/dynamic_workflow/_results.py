"""What `run_workflow` returns: results, bounded previews of completed calls, and terminal budget results."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic_ai.exceptions import ModelRetry

try:
    from pydantic_monty import MontyCrashedError
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'pydantic-monty is required for DynamicWorkflow. '
        'Install it with: uv add "pydantic-ai-harness[dynamic-workflow]"'
    ) from _import_error

_MAX_COMPLETED_DISPATCHES = 20
_MAX_TASK_PREVIEW_CHARS = 120
_MAX_RESULT_PREVIEW_CHARS = 300
_TRUNCATED_MARKER = ' ... [truncated]'


class BudgetExhausted(RuntimeError):
    """The run's `max_agent_calls` budget is spent; no further sub-agent runs are allowed."""

    def __init__(self, max_agent_calls: int) -> None:
        super().__init__(f'sub-agent call budget ({max_agent_calls}) exhausted')


def _workflow_result(result: object, printed: str) -> object:
    """Shape the tool return: the script's result, its captured `print()` output, or both."""
    if not printed:
        return result if result is not None else {}
    if result is None:
        return {'output': printed}
    return {'output': printed, 'result': result}


@dataclass(frozen=True)
class CompletedDispatch:
    """One sub-agent result completed by the failed workflow script."""

    agent_name: str
    task: str
    result: object


def _truncate_preview(value: str, max_chars: int) -> str:
    """Trim a preview with an explicit marker so the model can see it is incomplete."""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - len(_TRUNCATED_MARKER)] + _TRUNCATED_MARKER


def _json_preview(value: object, max_chars: int) -> str:
    """Render a compact JSON preview of a completed sub-agent result."""
    rendered = json.dumps(value, ensure_ascii=True, separators=(',', ':'))
    return _truncate_preview(rendered, max_chars)


def _completed_dispatch_lines(completed: list[CompletedDispatch]) -> list[str]:
    """Format the most recent completed sub-agent results for model-facing salvage."""
    shown = completed[-_MAX_COMPLETED_DISPATCHES:]
    lines = [
        f'{entry.agent_name}(task={json.dumps(_truncate_preview(entry.task, _MAX_TASK_PREVIEW_CHARS), ensure_ascii=True)})'
        f' -> {_json_preview(entry.result, _MAX_RESULT_PREVIEW_CHARS)}'
        for entry in shown
    ]
    omitted = len(completed) - len(shown)
    if omitted:
        lines.insert(0, f'... {omitted} earlier completed result(s) omitted ...')
    return lines


def completed_retry_section(completed: list[CompletedDispatch]) -> str:
    """Build the optional retry-message section listing salvageable completed results."""
    lines = _completed_dispatch_lines(completed)
    if not lines:
        return ''
    listing = '\n'.join(f'- {line}' for line in lines)
    return (
        '\n\nCompleted sub-agent results from the failed script '
        '(up to 20 bounded previews; reuse untruncated values instead of re-calling them; '
        'their budget was already spent):\n'
        f'{listing}'
    )


def budget_terminal_result(
    *,
    max_agent_calls: int,
    last_error: str,
    completed_dispatches: list[CompletedDispatch],
) -> dict[str, object]:
    """Build the terminal result returned after the exact sub-agent-call budget is exhausted."""
    return {
        'error': (
            f'This run exhausted its sub-agent call budget ({max_agent_calls}). '
            'Conclude using the results already gathered; further sub-agent calls in '
            'this run will be refused.'
        ),
        'last_error': last_error,
        'completed': _completed_dispatch_lines(completed_dispatches),
    }


def worker_crash_result(
    *,
    crash: MontyCrashedError,
    budget_exhausted: bool,
    max_agent_calls: int,
    completed_dispatches: list[CompletedDispatch],
) -> dict[str, object]:
    if budget_exhausted:
        return budget_terminal_result(
            max_agent_calls=max_agent_calls,
            last_error='The workflow script crashed the sandbox worker after exhausting the sub-agent budget.',
            completed_dispatches=completed_dispatches,
        )
    raise ModelRetry(
        'The workflow script crashed the sandbox worker. Revise the script and try again.'
        f'{completed_retry_section(completed_dispatches)}'
    ) from crash


def completed_workflow_result(
    *,
    completed_output: object,
    printed: str,
    budget_exhausted: bool,
    max_agent_calls: int,
    completed_dispatches: list[CompletedDispatch],
) -> object:
    """Return a normal result unless the script caught a terminal budget error."""
    if budget_exhausted:
        return budget_terminal_result(
            max_agent_calls=max_agent_calls,
            last_error='The workflow caught the budget error and completed after exhausting the sub-agent budget.',
            completed_dispatches=completed_dispatches,
        )
    return _workflow_result(completed_output, printed)
