from __future__ import annotations, annotations as _annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter
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

# AgentWorker output type needs to be invariant for use in both parameter and return positions
WorkerOutputT = TypeVar('WorkerOutputT')

try:
    from starlette.middleware import Middleware
    from starlette.routing import Route
    from starlette.types import ExceptionHandler, Lifespan

    from fasta2a.applications import FastA2A
    from fasta2a.broker import Broker, InMemoryBroker
    from fasta2a.schema import (
        Artifact,
        DataPart,
        Message,
        Part,
        Provider,
        Skill,
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
    worker = AgentWorker(agent=agent, broker=broker, storage=storage)

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
# Generic parameters are reversed compared to Agent because AgentDepsT has a default
class AgentWorker(Worker, Generic[WorkerOutputT, AgentDepsT]):
    """A worker that uses an agent to execute tasks."""

    agent: Agent[AgentDepsT, WorkerOutputT]

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'], history_length=params.get('history_length'))
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')
        if 'context_id' not in task:
            raise ValueError('Task must have a context_id')

        task_id = task['id']
        context_id = task['context_id']

        try:
            await self.storage.update_task(task_id, state='working')

            # TODO(Marcelo): We need to have a way to communicate when the task is set to `input-required`. Maybe
            # a custom `output_type` with a `more_info_required` field, or something like that.

            history = task.get('history', [])
            message_history = self.build_message_history(history)

            # TODO(Marcelo): We need to make this more customizable e.g. pass deps.
            result = await self.agent.run(message_history=message_history)  # type: ignore

            # Create both a message and artifact for the result
            # This ensures the complete conversation is preserved in history while
            # also marking the output as a durable artifact
            message_id = str(uuid.uuid4())

            # Create message parts based on output type
            message_part = self._convert_result_to_part(result.output)
            message_parts: list[Part] = [message_part]

            # Add result as a message to preserve conversation flow
            result_message = Message(
                role='agent',
                parts=message_parts,
                kind='message',
                message_id=message_id,
                task_id=task_id,
                context_id=context_id,
            )
            await self.storage.add_message(result_message)

            # Also create artifacts for durable outputs
            artifacts = self.build_artifacts(result.output)
            await self.storage.update_task(task_id, state='completed', artifacts=artifacts)

        except Exception:
            # Ensure task is marked as failed on any error
            await self.storage.update_task(task_id, state='failed')
            raise  # Re-raise to maintain error visibility

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_artifacts(self, result: WorkerOutputT) -> list[Artifact]:
        """Build artifacts from agent result.

        All agent outputs become artifacts to mark them as durable task outputs.
        For string results, we use TextPart. For structured data, we use DataPart.
        Metadata is included to preserve type information.
        """
        artifact_id = str(uuid.uuid4())
        part = self._convert_result_to_part(result)
        metadata = self._build_result_metadata(result)
        return [Artifact(artifact_id=artifact_id, name='result', parts=[part], metadata=metadata)]

    def _convert_result_to_part(self, result: WorkerOutputT) -> Part:
        """Convert agent result to a Part (TextPart or DataPart).

        For string results, returns a TextPart.
        For structured data, returns a DataPart with properly serialized data.
        """
        if isinstance(result, str):
            return A2ATextPart(kind='text', text=result)
        else:
            # For structured data, create a DataPart
            try:
                # Try using TypeAdapter for proper serialization
                output_type = type(result)
                type_adapter: TypeAdapter[WorkerOutputT] = TypeAdapter(output_type)
                data = type_adapter.dump_python(result, mode='json')
            except Exception:
                # Fallback for types that TypeAdapter can't handle
                if is_dataclass(result) and not isinstance(result, type):
                    data = asdict(result)
                else:
                    # Last resort - convert to string
                    data = str(result)

            return DataPart(kind='data', data=data)

    def _build_result_metadata(self, result: WorkerOutputT) -> dict[str, Any]:
        """Build metadata for the result artifact.

        Captures type information and JSON schema when available.
        """
        metadata: dict[str, Any] = {
            'type': type(result).__name__,
        }

        # For non-string types, attempt to capture JSON schema
        if not isinstance(result, str):
            output_type = type(result)
            type_adapter: TypeAdapter[WorkerOutputT] = TypeAdapter(output_type)
            try:
                metadata['json_schema'] = type_adapter.json_schema()
            except Exception:
                # Some types don't support JSON schema generation
                pass

        return metadata

    def build_message_history(self, history: list[Message]) -> list[ModelMessage]:
        model_messages: list[ModelMessage] = []
        for message in history:
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
                file = part['file']
                if 'data' in file:
                    data = file['data'].encode('utf-8')
                    content = BinaryContent(data=data, media_type=file['mime_type'])
                    model_parts.append(UserPromptPart(content=[content]))
                elif 'uri' in file:
                    uri = file['uri']
                    mime_type = file.get('mime_type', 'application/octet-stream')
                    for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
                        content = url_cls(url=uri)
                        try:
                            content.media_type
                        except ValueError:  # pragma: no cover
                            continue
                        else:
                            break
                    else:
                        raise ValueError(f'Unknown file type: {mime_type}')  # pragma: no cover
                    model_parts.append(UserPromptPart(content=[content]))
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
