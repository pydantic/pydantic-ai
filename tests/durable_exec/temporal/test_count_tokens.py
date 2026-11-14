"""The token count `UsageLimits(count_tokens_before_request=True)` asks for, under Temporal.

The count is provider I/O, so it runs in an activity of its own ahead of the request, and the workflow sees what the
activity returned: the usage, the framework metadata the count wrote on the request, or the error it raised. Not VCR
tests: the behavior under test is which activities a workflow schedules and what its history replays, which only a
Temporal server shows.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits

try:
    from temporalio import activity, workflow
    from temporalio.client import Client, WorkflowHistory
    from temporalio.common import RetryPolicy
    from temporalio.exceptions import ActivityError, ApplicationError
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
    from temporalio.workflow import ActivityConfig

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-durability')]

_counted_in_activity: list[bool] = []
"""Whether each count ran in an activity. Activities run outside the workflow sandbox, so they append to this module."""


def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('ok')])


class _CountingModel(FunctionModel):
    """Counts without I/O and records a framework key on the counted request, as `AnthropicModel` does after a retry."""

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        _counted_in_activity.append(activity.in_activity())
        request = next(message for message in reversed(messages) if isinstance(message, ModelRequest))
        request.metadata = {'__pydantic_ai__': {'counted': True}}
        return RequestUsage(input_tokens=5)


_counting_agent = Agent(
    _CountingModel(_reply),
    name='count_tokens',
    capabilities=[TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class CountTokensWorkflow:
    @workflow.run
    async def run(self, input_tokens_limit: int) -> tuple[str, list[dict[str, Any] | None]]:
        limits = UsageLimits(input_tokens_limit=input_tokens_limit, count_tokens_before_request=True)
        result = await _counting_agent.run('hi', usage_limits=limits)
        return result.output, [
            message.metadata for message in result.all_messages() if isinstance(message, ModelRequest)
        ]


def _scheduled_activities(history: WorkflowHistory) -> list[str]:
    return [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]


async def test_count_tokens_runs_in_its_own_activity(client: Client):
    """The count is an activity ahead of the request, and the key it wrote on its deserialized request reaches the run."""
    _counted_in_activity.clear()
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[CountTokensWorkflow], plugins=[AgentPlugin(_counting_agent)]
    ):
        handle = await client.start_workflow(
            CountTokensWorkflow.run,
            args=[100],
            id=CountTokensWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        output = await handle.result()

    assert output == ('ok', [{'__pydantic_ai__': {'counted': True}}])
    assert _counted_in_activity == [True]
    assert _scheduled_activities(await handle.fetch_history()) == [
        'agent__count_tokens__model_count_tokens',
        'agent__count_tokens__model_request',
    ]


async def test_count_tokens_replays_history_recorded_with_a_workflow_side_count(
    client: Client, monkeypatch: pytest.MonkeyPatch
):
    """A history recorded when the count ran in workflow code replays without scheduling a count activity.

    Unsandboxed, so the module-level list sees where the count ran in both the original run and the replay.
    """
    _counted_in_activity.clear()

    # Without the patch marker, the workflow runs the way it did before: the count runs in workflow code.
    def unpatched(patch_id: str) -> bool:
        return False

    monkeypatch.setattr(workflow, 'patched', unpatched)
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CountTokensWorkflow],
        plugins=[AgentPlugin(_counting_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            CountTokensWorkflow.run,
            args=[100],
            id=f'{CountTokensWorkflow.__name__}-legacy',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        assert await handle.result() == ('ok', [{'__pydantic_ai__': {'counted': True}}])
    monkeypatch.undo()

    history = await handle.fetch_history()
    assert _scheduled_activities(history) == ['agent__count_tokens__model_request']
    replay = await Replayer(
        workflows=[CountTokensWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=UnsandboxedWorkflowRunner()
    ).replay_workflow(history)
    assert replay.replay_failure is None
    assert _counted_in_activity == [False, False]


_count_attempts: list[str] = []


class _CannotCountModel(FunctionModel):
    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        _count_attempts.append('not_implemented')
        return await super().count_tokens(messages, model_settings, model_request_parameters)


class _CountConfigError(UserError):
    """An application's own `UserError`, which fails the count's activity as it would fail a model request's."""


class _RejectedCountModel(FunctionModel):
    kind = 'user_error'
    error: UserError = UserError('This request cannot be counted.')

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        _count_attempts.append(self.kind)
        raise self.error


class _MisconfiguredCountModel(_RejectedCountModel):
    kind = 'user_error_subclass'
    error = _CountConfigError('The app is misconfigured.')


# Retries allowed, so a count error that Temporal retried would show up as a second attempt.
_RETRYING_ACTIVITY_CONFIG = ActivityConfig(
    start_to_close_timeout=timedelta(seconds=60),
    retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(milliseconds=1)),
)
_cannot_count_agent = Agent(
    _CannotCountModel(_reply),
    name='count_tokens_not_implemented',
    capabilities=[TemporalDurability(activity_config=_RETRYING_ACTIVITY_CONFIG)],
)
_rejected_count_agent = Agent(
    _RejectedCountModel(_reply),
    name='count_tokens_user_error',
    capabilities=[TemporalDurability(activity_config=_RETRYING_ACTIVITY_CONFIG)],
)
_misconfigured_count_agent = Agent(
    _MisconfiguredCountModel(_reply),
    name='count_tokens_user_error_subclass',
    capabilities=[TemporalDurability(activity_config=_RETRYING_ACTIVITY_CONFIG)],
)
_count_error_agents = {
    'not_implemented': _cannot_count_agent,
    'user_error': _rejected_count_agent,
    'user_error_subclass': _misconfigured_count_agent,
}


@workflow.defn
class CountTokensErrorWorkflow:
    @workflow.run
    async def run(self, error: str) -> str:
        try:
            await _count_error_agents[error].run('hi', usage_limits=UsageLimits(count_tokens_before_request=True))
        except NotImplementedError as e:
            return f'NotImplementedError: {e}'
        except UserError as e:
            return f'UserError: {e}'
        except ActivityError as e:
            assert isinstance(e.cause, ApplicationError)
            return f'ActivityError: {e.cause.type}: {e.cause.message}'
        return 'counted'  # pragma: no cover


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (
            'not_implemented',
            'NotImplementedError: Token counting ahead of the request is not supported by _CannotCountModel',
        ),
        ('user_error', 'UserError: This request cannot be counted.'),
        ('user_error_subclass', 'ActivityError: _CountConfigError: The app is misconfigured.'),
    ],
)
async def test_count_tokens_errors_fail_the_count_once(client: Client, error: str, expected: str):
    """A model that can't count, or a request it rejects, fails the count once, with the error the model raised.

    `NotImplementedError` is what Pydantic AI Harness falls back on when a nested agent's model can't count. A
    `UserError` subclass carries more than its message, so it fails the activity as it would fail a model request's.
    """
    _count_attempts.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[CountTokensErrorWorkflow],
        plugins=[AgentPlugin(agent) for agent in _count_error_agents.values()],
    ):
        output = await client.execute_workflow(
            CountTokensErrorWorkflow.run,
            args=[error],
            id=f'{CountTokensErrorWorkflow.__name__}-{error}',
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
    assert output == expected
    assert _count_attempts == [error]
