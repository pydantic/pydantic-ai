from __future__ import annotations, annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Generic

from typing_extensions import assert_never

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    UserPromptPart,
    VideoUrl,
)

from .agent import Agent, AgentDepsT, OutputDataT

try:
    from starlette.middleware import Middleware
    from starlette.routing import Route
    from starlette.types import ExceptionHandler, Lifespan

    from fasta2a.applications import FastA2A
    from fasta2a.broker import Broker, InMemoryBroker
    from fasta2a.schema import (
        Artifact,
        Message,
        Part,
        Provider,
        Skill,
        Task,
        TaskIdParams,
        TaskSendParams,
        TextPart as A2ATextPart,
    )
    from fasta2a.storage import InMemoryStorage, Storage
    from fasta2a.worker import Worker
except ImportError as _import_error:
    raise ImportError(
        'Please install the `fasta2a` package to use `Agent.to_a2a()` method, '
        'you can use the `a2a` optional group — `pip install "pydantic-ai-slim[a2a]"`'
    ) from _import_error


@asynccontextmanager
async def worker_lifespan(app: FastA2A, worker: Worker) -> AsyncIterator[None]:
    """Custom lifespan that runs the worker during application startup.

    This ensures the worker is started and ready to process tasks as soon as the application starts.
    """
    async with app.task_manager:
        async with worker.run():
            yield


def agent_to_a2a(
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    deps_factory: Callable[[Task], AgentDepsT] | None = None,
    storage: Storage | None = None,
    broker: Broker | None = None,
    # Agent card
    name: str | None = None,
    url: str = 'http://localhost:8000',
    version: str = '1.0.0',
    description: str | None = None,
    provider: Provider | None = None,
    skills: list[Skill] | None = None,
    # Starlette
    debug: bool = False,
    routes: Sequence[Route] | None = None,
    middleware: Sequence[Middleware] | None = None,
    exception_handlers: dict[Any, ExceptionHandler] | None = None,
    lifespan: Lifespan[FastA2A] | None = None,
) -> FastA2A:
    """Create a FastA2A server from an agent."""
    storage = storage or InMemoryStorage()
    broker = broker or InMemoryBroker()
    worker = AgentWorker(agent=agent, broker=broker, storage=storage, deps_factory=deps_factory)

    lifespan = lifespan or partial(worker_lifespan, worker=worker)

    return FastA2A(
        storage=storage,
        broker=broker,
        name=name or agent.name,
        url=url,
        version=version,
        description=description,
        provider=provider,
        skills=skills,
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )


@dataclass
class AgentWorker(Worker, Generic[AgentDepsT, OutputDataT]):
    """A worker that uses an agent to execute tasks."""

    agent: Agent[AgentDepsT, OutputDataT]
    deps_factory: Callable[[Task], AgentDepsT] | None = None

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'], history_length=params.get('history_length'))
        assert task is not None, f'Task {params["id"]} not found'
        assert 'session_id' in task, 'Task must have a session_id'

        await self.storage.update_task(task['id'], state='working')

        # TODO(Marcelo): We need to have a way to communicate when the task is set to `input-required`. Maybe
        # a custom `output_type` with a `more_info_required` field, or something like that.

        task_history = task.get('history', [])
        message_history = self.build_message_history(task_history=task_history)

        # Initialize dependencies if factory provided
        if self.deps_factory is not None:
            deps = self.deps_factory(task)
            result = await self.agent.run(message_history=message_history, deps=deps)
        else:
            # No deps_factory provided - this only works if the agent accepts None for deps
            # (e.g., Agent[None, ...] or Agent[Optional[...], ...])
            # If the agent requires deps, this will raise TypeError at runtime
            result = await self.agent.run(message_history=message_history)  # type: ignore[call-arg]

        artifacts = self.build_artifacts(result.output)
        await self.storage.update_task(task['id'], state='completed', artifacts=artifacts)

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_artifacts(self, result: Any) -> list[Artifact]:
        # TODO(Marcelo): We need to send the json schema of the result on the metadata of the message.
        return [Artifact(name='result', index=0, parts=[A2ATextPart(kind='text', text=str(result))])]

    def build_message_history(self, task_history: list[Message]) -> list[ModelMessage]:
        model_messages: list[ModelMessage] = []
        for message in task_history:
            if message['role'] == 'user':
                model_messages.append(ModelRequest(parts=self._map_request_parts(message['parts'])))
            else:
                model_messages.append(ModelResponse(parts=self._map_response_parts(message['parts'])))
        return model_messages

    def _map_request_parts(self, parts: list[Part]) -> list[ModelRequestPart]:
        model_parts: list[ModelRequestPart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(UserPromptPart(content=part['text']))
            elif part['kind'] == 'file':
                if 'data' in part:
                    data = part['data'].encode('utf-8')
                    mime_type = part.get('mime_type', 'application/octet-stream')
                    content = BinaryContent(data=data, media_type=mime_type)
                    model_parts.append(UserPromptPart(content=[content]))
                elif 'uri' in part:
                    url = part['uri']
                    mime_type = part.get('mime_type', '')
                    if mime_type.startswith('image/'):
                        content = ImageUrl(url=url)
                    elif mime_type.startswith('audio/'):
                        content = AudioUrl(url=url)
                    elif mime_type.startswith('video/'):
                        content = VideoUrl(url=url)
                    else:
                        content = DocumentUrl(url=url)
                    model_parts.append(UserPromptPart(content=[content]))
                else:
                    raise ValueError('FilePart must have either data or uri')
            elif part['kind'] == 'data':
                # TODO(Marcelo): Maybe we should use this for `ToolReturnPart`, and `RetryPromptPart`.
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts

    def _map_response_parts(self, parts: list[Part]) -> list[ModelResponsePart]:
        model_parts: list[ModelResponsePart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(TextPart(content=part['text']))
            elif part['kind'] == 'file':  # pragma: no cover
                raise NotImplementedError('File parts are not supported yet.')
            elif part['kind'] == 'data':  # pragma: no cover
                raise NotImplementedError('Data parts are not supported yet.')
            else:  # pragma: no cover
                assert_never(part)
        return model_parts
