from __future__ import annotations, annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Generic

from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.types import ExceptionHandler, Lifespan
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
    from fasta2a.applications import FastA2A
    from fasta2a.runner import Runner
    from fasta2a.schema import Artifact, Message, Part, Provider, Skill, TaskSendParams, TextPart as A2ATextPart
    from fasta2a.storage import InMemoryStorage, Storage
    from fasta2a.worker import InMemoryWorker, TaskContext, Worker
except ImportError as _import_error:
    raise ImportError(
        'Please install the `fasta2a` package to use `Agent.to_a2a()` method, '
        'you can use the `a2a` optional group — `pip install "pydantic-ai-slim[a2a]"`'
    ) from _import_error


def agent_to_a2a(
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    storage: Storage | None = None,
    worker: Worker | None = None,
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
    worker = worker or InMemoryWorker(runner=AgentRunner(agent=agent), storage=storage)
    return FastA2A(
        storage=storage,
        worker=worker,
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
class AgentRunner(Runner, Generic[AgentDepsT, OutputDataT]):
    """A runner that uses an agent to execute tasks."""

    agent: Agent[AgentDepsT, OutputDataT]

    async def run(self, task_ctx: TaskContext[TaskSendParams]) -> None:
        params = task_ctx.params
        task = await task_ctx.storage.load_task(params['id'], history_length=params.get('history_length'))
        assert task is not None, f'Task {task_ctx.params["id"]} not found'
        assert 'session_id' in task, 'Task must have a session_id'

        await task_ctx.storage.update_task(task['id'], state='working')

        task_history = task.get('history', [])
        task_history.append(params['message'])
        message_history = self.build_message_history(task_history=task_history)

        # TODO(Marcelo): We need to have a way to communicate when the task is set to `input-required`. Maybe
        # a custom `output_type` with a `more_info_required` field, or something like that.

        # TODO(Marcelo): We need to make this more customizable e.g. pass deps.
        result = await self.agent.run(message_history=message_history)  # type: ignore

        artifacts = self.build_artifacts(result.output)
        await task_ctx.storage.update_task(task['id'], state='completed', artifacts=artifacts)

    def build_artifacts(self, result: Any) -> list[Artifact]:
        # TODO(Marcelo): We need to send the json schema of the result on the metadata of the message.
        return [Artifact(name='result', index=0, parts=[A2ATextPart(type='text', text=str(result))])]

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
            if part['type'] == 'text':
                model_parts.append(UserPromptPart(content=part['text']))
            elif part['type'] == 'file':
                file = part['file']
                if 'data' in file:
                    data = file['data'].encode('utf-8')
                    content = BinaryContent(data=data, media_type=file['mime_type'])
                    model_parts.append(UserPromptPart(content=[content]))
                else:
                    url = file['url']
                    for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
                        content = url_cls(url=url)
                        try:
                            content.media_type
                        except ValueError:
                            continue
                        else:
                            break
                    else:
                        raise ValueError(f'Unknown file type: {file["mime_type"]}')
                    model_parts.append(UserPromptPart(content=[content]))
            elif part['type'] == 'data':
                # TODO(Marcelo): Maybe we should use this for `ToolReturnPart`, and `RetryPromptPart`.
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts

    def _map_response_parts(self, parts: list[Part]) -> list[ModelResponsePart]:
        model_parts: list[ModelResponsePart] = []
        for part in parts:
            if part['type'] == 'text':
                model_parts.append(TextPart(content=part['text']))
            elif part['type'] == 'file':
                raise NotImplementedError('File parts are not supported yet.')
            elif part['type'] == 'data':
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts
