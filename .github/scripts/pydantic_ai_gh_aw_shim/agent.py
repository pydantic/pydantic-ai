"""Repository workflow policy composed with the public CLAI 2 engine's Coder."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from fnmatch import fnmatchcase
from functools import partial

import logfire
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext, UsageLimitExceeded
from pydantic_ai.capabilities import (
    AbstractCapability,
    LocalWorkspace,
    PrepareTools,
    ProcessHistory,
    ResolveModelId,
    WrapRunHandler,
)
from pydantic_ai.models import Model, ModelResolutionContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import Tool
from pydantic_ai_harness import Coder, RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

from . import cli
from .shared import reset_context_state

result_metadata: dict[str, object] = {}


@dataclass
class WorkflowPolicy(AbstractCapability[object]):
    """Keep workflow budgets and terminal health metadata around CLAI's run."""

    timed_out: bool = False

    async def for_run(self, ctx: RunContext[object]) -> WorkflowPolicy:
        return WorkflowPolicy()

    def expire(self, ctx: RunContext[object]) -> None:
        """Cancel the run owner, rather than its capability lifecycle waiter."""
        self.timed_out = True
        ctx.cancel()

    async def on_run_error(self, ctx: RunContext[object], *, error: BaseException) -> AgentRunResult[object]:
        if self.timed_out:
            raise cli.RunTimeLimitExceeded('Workflow exceeded its run budget') from error
        raise error

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        reset_context_state()
        result_metadata.clear()
        budget = cli.run_timeout_secs()
        cli.run_deadline = time.monotonic() + budget
        cap = cli.wall_clock_caps.set((*cli.wall_clock_caps.get(), lambda: self.timed_out))
        started = time.monotonic()
        error: BaseException | None = None
        timer = asyncio.get_running_loop().call_later(budget, self.expire, ctx)
        try:
            async with AsyncExitStack() as stack:
                if _model is not None:
                    await stack.enter_async_context(_model.client)
                return await handler()
        except UsageLimitExceeded as exc:
            if cli.safe_output_pending():
                error = exc
                raise
            return TypeAdapter(AgentRunResult[object]).validate_python(
                {
                    'output': 'Request limit reached after the safe output was emitted',
                    'messages': ctx.messages,
                    'usage': ctx.usage,
                    'run_id': ctx.run_id,
                    'conversation_id': ctx.conversation_id,
                }
            )
        except BaseException as exc:
            error = exc
            raise
        finally:
            timer.cancel()
            cli.run_deadline = None
            cli.wall_clock_caps.reset(cap)
            run_attempt: int | None = None
            try:
                parsed_attempt = int(os.environ.get('PYDANTIC_AI_RUN_ATTEMPT', ''))
            except ValueError:
                pass
            else:
                if parsed_attempt > 0:
                    run_attempt = parsed_attempt
            health: dict[str, object] = {
                'workflow': os.environ.get('GITHUB_WORKFLOW'),
                'task_key': os.environ.get('PYDANTIC_AI_TASK_ID') or os.environ.get('PYDANTIC_AI_TASK_KEY'),
                'trigger_event': os.environ.get('PYDANTIC_AI_TRIGGER_EVENT'),
                'run_attempt': run_attempt,
            }
            if error is not None and not isinstance(error, asyncio.CancelledError):
                health['failure'] = cli.failure_details(error)
            result_metadata['provider_health'] = health
            # Retain the legacy launcher's DTO; protected capture authenticates the
            # same value through PAI_RESULT_METADATA in the canonical terminal record.
            print(json.dumps({'type': 'result', 'provider_health': health}), file=sys.stderr, flush=True)
            logfire.info(
                'Workflow finished',
                task_key=health['task_key'],
                outcome='cancelled' if isinstance(error, asyncio.CancelledError) else 'error' if error else 'success',
                duration_seconds=time.monotonic() - started,
                requests=ctx.usage.requests,
                input_tokens=ctx.usage.input_tokens,
                output_tokens=ctx.usage.output_tokens,
                cache_read_tokens=ctx.usage.cache_read_tokens,
                cache_write_tokens=ctx.usage.cache_write_tokens,
            )


_model: AnthropicModel | None = None


def workflow_model(ctx: ModelResolutionContext[object], model_id: str) -> Model:
    """Resolve CLAI's explicit model with the workflow's existing transport policy."""
    del ctx
    global _model
    provider, separator, name = model_id.partition(':')
    if provider != 'anthropic' or not separator or not name:
        raise ValueError(f'The Z.ai workflow expects an Anthropic model ID, got {model_id!r}')
    if _model is None:
        model, _ = cli.build_model(cli.Args(model=name))
        assert isinstance(model, AnthropicModel)
        _model = model
    if _model.model_name != name:
        raise ValueError('A workflow cannot change its configured model during execution')
    return _model


def build_agent() -> Agent[None, str]:
    """Compose Coder with the repository workflow policies."""
    attention = os.environ.get('GITHUB_WORKFLOW') == cli.ATTENTION_WORKFLOW
    env: dict[str, str] = {
        name: value
        for name, value in os.environ.items()
        if not any(fnmatchcase(name, pattern) for pattern in LLM_API_KEY_ENV_PATTERNS)
        and name != 'GH_AW_SESSION_FRAME_KEY'
    }
    agent: Agent[None, str] = Agent(
        name='github-workflow',
        instructions=[cli.request_budget_notice, cli.time_budget_notice],
        retries={'output': cli.NO_SAFE_OUTPUT_RETRIES},
        tools=[] if attention else [Tool(cli.task, name='Task')],
        capabilities=[
            ResolveModelId(workflow_model),
            WorkflowPolicy(),
            # Inside `WorkflowPolicy`, whose run budget it reads through `wall_clock_caps`.
            cli.MarkTimedOutSpans(),
            cli.RecoverMCPToolErrors(),
            PrepareTools(cli.hold_time_reserve_for_safe_output),
            ProcessHistory(partial(cli.compact_history, preserve_prompt=True)),
            LocalWorkspace('.', env=env, read_only=attention),
            # Checkout guidance accompanies file traversal, as in the read-only
            # delegates; it must not become the reviewer's system instructions.
            *(
                []
                if attention
                else [RepoContext(autoload_instructions=False, nested_traversal=True, nested_inject='contents')]
            ),
            Coder(sub_agents=False, caching=False, repo_context=False),
        ],
    )
    agent.output_validator(cli.require_safe_output)
    return agent


agent = build_agent()

if os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT') or os.environ.get('PAI_OTEL_EXPORT_PATH'):
    # As `cli.configure_observability`: request bodies would repeat each `chat` span's input messages.
    logfire.instrument_httpx(capture_headers=True, capture_response_body=True)
    logfire.instrument_mcp()
