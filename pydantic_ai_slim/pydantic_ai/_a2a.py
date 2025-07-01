from __future__ import annotations, annotations as _annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from typing import Any, Callable, Generic, cast

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
    ThinkingPart,
    ToolCallPart,
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
        DataPart,
        Message,
        Part,
        Provider,
        Skill,
        Task,
        TaskArtifactUpdateEvent,
        TaskIdParams,
        TaskSendParams,
        TaskStatusUpdateEvent,
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
        assert 'context_id' in task, 'Task must have a context_id'

        task_id = task['id']
        context_id = task['context_id']

        # Update storage and send working status event
        await self.storage.update_task(task_id, state='working')
        await self.broker.send_stream_event(
            task_id,
            TaskStatusUpdateEvent(
                task_id=task_id, context_id=context_id, kind='status-update', status={'state': 'working'}, final=False
            ),
        )

        # TODO(Marcelo): We need to have a way to communicate when the task is set to `input-required`. Maybe
        # a custom `output_type` with a `more_info_required` field, or something like that.

        try:
            context_history = await self.storage.get_context_history(
                context_id, history_length=params.get('history_length')
            )
            message_history = self.build_message_history(context_history)
            assert len(message_history) and isinstance(message_history[-1], ModelRequest)
            # Extract text content from the last message's parts
            text_parts: list[str] = []
            for part in message_history[-1].parts:
                if hasattr(part, 'content'):
                    if isinstance(part.content, str):
                        text_parts.append(part.content)
            current_message: str = ''.join(text_parts)
            message_history = message_history[:-1]

            # Initialize dependencies if factory provided
            deps: AgentDepsT = cast(AgentDepsT, self.deps_factory(task) if self.deps_factory is not None else None)

            async with self.agent.iter(current_message, message_history=message_history, deps=deps) as run:
                message_id = str(uuid.uuid4())
                node = run.next_node
                while not self.agent.is_end_node(node):
                    # Check if this node has a model response
                    if hasattr(node, 'model_response'):
                        model_response = getattr(node, 'model_response')
                        # Convert model response parts to A2A parts
                        a2a_parts = self._response_parts_to_a2a(model_response.parts)

                        if a2a_parts:
                            # Send incremental message event
                            incremental_message = Message(
                                role='agent',
                                parts=a2a_parts,
                                kind='message',
                                message_id=message_id,
                                task_id=task_id,
                                context_id=context_id,
                            )
                            await self.storage.add_message(incremental_message)
                            await self.broker.send_stream_event(task_id, incremental_message)

                    # Move to next node
                    current = node
                    node = await run.next(current)

                # Run finished - get the final result
                if run.result is None:
                    raise RuntimeError('Agent finished without producing a result')

                artifacts: list[Artifact] = []
                if isinstance(run.result.output, str):
                    final_message = Message(
                        role='agent',
                        parts=[A2ATextPart(kind='text', text=run.result.output)],
                        kind='message',
                        message_id=message_id,
                        task_id=task_id,
                        context_id=context_id,
                    )
                    await self.storage.add_message(final_message)
                    await self.broker.send_stream_event(task_id, final_message)
                else:
                    # Create artifact for non-string outputs
                    artifact_id = str(uuid.uuid4())
                    output: OutputDataT = run.result.output
                    metadata: dict[str, Any] = {'type': type(output).__name__}

                    try:
                        # Create TypeAdapter for the output type
                        output_type = type(output)
                        type_adapter: TypeAdapter[OutputDataT] = TypeAdapter(output_type)

                        # Serialize to Python dict/list for DataPart
                        data = type_adapter.dump_python(output, mode='json')

                        # Get JSON schema if possible
                        try:
                            json_schema = type_adapter.json_schema()
                            metadata['json_schema'] = json_schema
                            if hasattr(output, '__class__'):
                                metadata['class_name'] = output.__class__.__name__
                        except Exception:
                            raise
                            # Some types may not support JSON schema generation
                            pass

                    except Exception:
                        raise
                        # Fallback for types that TypeAdapter can't handle
                        if is_dataclass(output):
                            data = asdict(output)  # type: ignore[arg-type]
                            metadata['type'] = 'dataclass'
                            metadata['class_name'] = output.__class__.__name__
                        else:
                            # Last resort - convert to string
                            data = str(output)
                            metadata['type'] = 'string_fallback'

                    # Create artifact with DataPart
                    artifact = Artifact(
                        artifact_id=artifact_id,
                        name='result',
                        parts=[DataPart(kind='data', data=data)],
                        metadata=metadata,
                    )
                    artifacts.append(artifact)

                    # Send artifact update event
                    await self.broker.send_stream_event(
                        task_id,
                        TaskArtifactUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            kind='artifact-update',
                            artifact=artifact,
                            last_chunk=True,
                        ),
                    )

            # Update storage and send completion event
            await self.storage.update_task(task_id, state='completed', artifacts=artifacts if artifacts else None)
            await self.broker.send_stream_event(
                task_id,
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    kind='status-update',
                    status={'state': 'completed'},
                    final=True,
                ),
            )
        except Exception:
            # Update storage and send failure event
            await self.storage.update_task(task_id, state='failed')
            await self.broker.send_stream_event(
                task_id,
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    kind='status-update',
                    status={'state': 'failed'},
                    final=True,
                ),
            )
            raise

    async def cancel_task(self, params: TaskIdParams) -> None:
        pass

    def build_artifacts(self, result: Any) -> list[Artifact]:
        # TODO(Marcelo): We need to send the json schema of the result on the metadata of the message.
        artifact_id = str(uuid.uuid4())
        return [Artifact(artifact_id=artifact_id, name='result', parts=[A2ATextPart(kind='text', text=str(result))])]

    def build_message_history(self, history: list[Message]) -> list[ModelMessage]:
        model_messages: list[ModelMessage] = []
        for message in history:
            if message['role'] == 'user':
                model_messages.append(ModelRequest(parts=self._request_parts_from_a2a(message['parts'])))
            else:
                model_messages.append(ModelResponse(parts=self._response_parts_from_a2a(message['parts'])))

        return model_messages

    def _request_parts_from_a2a(self, parts: list[Part]) -> list[ModelRequestPart]:
        """Convert A2A Part objects to pydantic-ai ModelRequestPart objects.

        This handles the conversion from A2A protocol parts (text, file, data) to
        pydantic-ai's internal request parts (UserPromptPart with various content types).

        Args:
            parts: List of A2A Part objects from incoming messages

        Returns:
            List of ModelRequestPart objects for the pydantic-ai agent
        """
        model_parts: list[ModelRequestPart] = []
        for part in parts:
            if part['kind'] == 'text':
                model_parts.append(UserPromptPart(content=part['text']))
            elif part['kind'] == 'file':
                file_content = part['file']
                if 'data' in file_content:
                    data = file_content['data'].encode('utf-8')
                    mime_type = file_content.get('mime_type', 'application/octet-stream')
                    content = BinaryContent(data=data, media_type=mime_type)
                    model_parts.append(UserPromptPart(content=[content]))
                elif 'uri' in file_content:
                    url = file_content['uri']
                    mime_type = file_content.get('mime_type', '')
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
                    raise ValueError('FilePart.file must have either data or uri')
            elif part['kind'] == 'data':
                # TODO(Marcelo): Maybe we should use this for `ToolReturnPart`, and `RetryPromptPart`.
                raise NotImplementedError('Data parts are not supported yet.')
            else:
                assert_never(part)
        return model_parts

    def _response_parts_from_a2a(self, parts: list[Part]) -> list[ModelResponsePart]:
        """Convert A2A Part objects to pydantic-ai ModelResponsePart objects.

        This handles the conversion from A2A protocol parts (text, file, data) to
        pydantic-ai's internal response parts. Currently only supports text parts
        as agent responses in A2A are expected to be text-based.

        Args:
            parts: List of A2A Part objects from stored agent messages

        Returns:
            List of ModelResponsePart objects for message history
        """
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

    def _response_parts_to_a2a(self, parts: list[ModelResponsePart]) -> list[Part]:
        """Convert pydantic-ai ModelResponsePart objects to A2A Part objects.

        This handles the conversion from pydantic-ai's internal response parts to
        A2A protocol parts. Different part types are handled as follows:
        - TextPart: Converted directly to A2A TextPart
        - ThinkingPart: Converted to TextPart with metadata indicating it's thinking
        - ToolCallPart: Skipped (internal to agent execution)

        Args:
            parts: List of ModelResponsePart objects from agent response

        Returns:
            List of A2A Part objects suitable for sending via A2A protocol
        """
        a2a_parts: list[Part] = []
        for part in parts:
            if isinstance(part, TextPart):
                if part.content:  # Only add non-empty text
                    a2a_parts.append(A2ATextPart(kind='text', text=part.content))
            elif isinstance(part, ThinkingPart):
                if part.content:  # Only add non-empty thinking
                    # Convert thinking to text with metadata
                    a2a_parts.append(
                        A2ATextPart(
                            kind='text',
                            text=part.content,
                            metadata={'type': 'thinking', 'thinking_id': part.id, 'signature': part.signature},
                        )
                    )
            elif isinstance(part, ToolCallPart):
                # Skip tool calls - they're internal to agent execution
                pass
        return a2a_parts
