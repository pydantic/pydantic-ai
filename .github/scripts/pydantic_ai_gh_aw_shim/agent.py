"""Repository workflow policy composed with the public CLAI 2 engine's Coder."""

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
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ProcessHistory, ResolveModelId, WrapRunHandler
from pydantic_ai.models import Model, ModelResolutionContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import Tool
from pydantic_ai_harness import Coder, RepoContext
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

from . import cli
from .shared import reset_context_state


@dataclass
class WorkflowPolicy(AbstractCapability[object]):
    """Keep workflow budgets and terminal health metadata around CLAI's run."""

    async def wrap_run(self, ctx: RunContext[object], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        reset_context_state()
        budget = cli.run_timeout_secs()
        cli.run_deadline = time.monotonic() + budget
        started = time.monotonic()
        error: BaseException | None = None
        try:
            async with AsyncExitStack() as stack:
                if _model is not None:
                    await stack.enter_async_context(_model.client)
                async with asyncio.timeout(budget):
                    return await handler()
        except TimeoutError as exc:
            error = cli.RunTimeLimitExceeded(f'Workflow exceeded its {budget}-second run budget')
            raise error from exc
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
            cli.run_deadline = None
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
            # The health monitor consumes this DTO from agent-stdio.log. The public
            # engine independently owns canonical session events and merged usage.
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
        tools=[Tool(cli.task, name='Task')],
        capabilities=[
            ResolveModelId(workflow_model),
            WorkflowPolicy(),
            cli.RecoverMCPToolErrors(),
            ProcessHistory(partial(cli.compact_history, preserve_prompt=True)),
            LocalWorkspace('.', env=env),
            # Checkout guidance accompanies file traversal, as in the read-only
            # delegates; it must not become the reviewer's system instructions.
            RepoContext(autoload_instructions=False, nested_traversal=True, nested_inject='contents'),
            Coder(sub_agents=False, caching=False, repo_context=False),
        ],
    )
    agent.output_validator(cli.require_safe_output)
    return agent


agent = build_agent()

if os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT'):
    logfire.instrument_httpx(capture_all=True)
    logfire.instrument_mcp()
