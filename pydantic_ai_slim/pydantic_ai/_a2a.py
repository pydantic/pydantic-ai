from __future__ import annotations, annotations as _annotations

import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
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
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
    VideoUrl,
)

from .agent import Agent, AgentDepsT, OutputDataT

# AgentWorker output type needs to be invariant for use in both parameter and return positions
WorkerOutputT = TypeVar('WorkerOutputT')

try:
    from fasta2a.applications import FastA2A
    from fasta2a.broker import Broker, InMemoryBroker
    from fasta2a.schema import (
        AgentProvider,
        Artifact,
        DataPart,
        Message,
        Part,
        Skill,
        TaskArtifactUpdateEvent,
        TaskIdParams,
        TaskSendParams,
        TaskStatusUpdateEvent,
        TextPart as A2ATextPart,
    )
    from fasta2a.storage import InMemoryStorage, Storage
    from fasta2a.worker import Worker
    from starlette.middleware import Middleware
    from starlette.routing import Route
    from starlette.types import ExceptionHandler, Lifespan
except ImportError as _import_error:
    raise ImportError(
        'Please install the `fasta2a` package to use `Agent.to_a2a()` method, '
        'you can use the `a2a` optional group — `pip install "pydantic-ai-slim[a2a]"`'
    ) from _import_error


@asynccontextmanager
async def worker_lifespan(app: FastA2A, worker: Worker, agent: Agent[AgentDepsT, OutputDataT]) -> AsyncIterator[None]:
    """Custom lifespan that runs the worker during application startup.

    This ensures the worker is started and ready to process tasks as soon as the application starts.
    """
    async with app.task_manager, agent:
        async with worker.run():
            yield


def agent_to_a2a(
    agent: Agent[AgentDepsT, OutputDataT],
    *,
    enable_streaming: bool = False,
    storage: Storage | None = None,
    broker: Broker | None = None,
    # Agent card
    name: str | None = None,
    url: str = 'http://localhost:8000',
    version: str = '1.0.0',
    description: str | None = None,
    provider: AgentProvider | None = None,
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
    worker = AgentWorker(agent=agent, broker=broker, storage=storage, enable_streaming=enable_streaming)

    lifespan = lifespan or partial(worker_lifespan, worker=worker, agent=agent)

    return FastA2A(
        storage=storage,
        broker=broker,
        name=name or agent.name,
        url=url,
        version=version,
        description=description,
        provider=provider,
        skills=skills,
        streaming=enable_streaming,
        debug=debug,
        routes=routes,
        middleware=middleware,
        exception_handlers=exception_handlers,
        lifespan=lifespan,
    )


@dataclass
class AgentWorker(Worker[list[ModelMessage]], Generic[WorkerOutputT, AgentDepsT]):
    """A worker that uses an agent to execute tasks."""

    agent: Agent[AgentDepsT, WorkerOutputT]
    enable_streaming: bool = False

    async def run_task(self, params: TaskSendParams) -> None:
        task = await self.storage.load_task(params['id'])
        if task is None:
            raise ValueError(f'Task {params["id"]} not found')  # pragma: no cover

        # TODO(Marcelo): Should we lock `run_task` on the `context_id`?
        # Ensure this task hasn't been run before
        if task['status']['state'] != 'submitted':
            raise ValueError(  # pragma: no cover
                f'Task {params["id"]} has already been processed (state: {task["status"]["state"]})'
            )

        await self.storage.update_task(task['id'], state='working')

        # Send working status streaming event
        await self.broker.send_stream_event(
            task['id'],
            TaskStatusUpdateEvent(
                task_id=task['id'],
                context_id=task['context_id'],
                kind='status-update',
                status={'state': 'working'},
                final=False,
            ),
        )

        # Load context - contains pydantic-ai message history from previous tasks in this conversation
        message_history = await self.storage.load_context(task['context_id']) or []
        message_history.extend(self.build_message_history(task.get('history', [])))

        try:
            # Stream processing with agent.iter()
            async with self.agent.iter(message_history=message_history, deps=None) as run:  # type: ignore
                node = run.next_node
                while not self.agent.is_end_node(node):
                    # Check if this node has a model response
                    if hasattr(node, 'model_response'):
                        model_response = getattr(node, 'model_response')
                        # Convert model response parts to A2A parts
                        a2a_parts = self._response_parts_to_a2a(model_response.parts)

                        if a2a_parts and self.enable_streaming:
                            # Send incremental message event with unique ID
                            incremental_message = Message(
                                role='agent',
                                parts=a2a_parts,
                                kind='message',
                                message_id=str(uuid.uuid4()),  # Generate unique ID per message
                            )
                            # Stream the incremental message
                            await self.broker.send_stream_event(task['id'], incremental_message)

                    # Move to next node
                    current = node
                    node = await run.next(current)

                    if self.enable_streaming:
                        # Update context with current messages after each step
                        await self.storage.update_context(task['context_id'], run.ctx.state.message_history)

                # Run finished - get the final result and update context with final messages
                assert run.result is not None  # Agent iteration should always produce a result
                await self.storage.update_context(task['context_id'], run.result.all_messages())

                # Convert new messages to A2A format for task history
                a2a_messages: list[Message] = []

                for message in run.result.new_messages():
                    if isinstance(message, ModelRequest):
                        # Skip user prompts - they're already in task history
                        continue
                    else:
                        # Convert response parts to A2A format
                        a2a_parts = self._response_parts_to_a2a(message.parts)
                        if a2a_parts:  # Add if there are visible parts (text/thinking)
                            a2a_messages.append(
                                Message(role='agent', parts=a2a_parts, kind='message', message_id=str(uuid.uuid4()))
                            )

                # Handle final result and create artifacts using build_artifacts method
                artifacts = self.build_artifacts(run.result.output)

                # Send artifact update events for all artifacts (only for structured outputs)
                if not isinstance(run.result.output, str):
                    for artifact in artifacts:
                        await self.broker.send_stream_event(
                            task['id'],
                            TaskArtifactUpdateEvent(
                                task_id=task['id'],
                                context_id=task['context_id'],
                                kind='artifact-update',
                                artifact=artifact,
                                last_chunk=True,
                            ),
                        )
        except Exception:
            await self.storage.update_task(task['id'], state='failed')

            # Send failure status streaming event
            await self.broker.send_stream_event(
                task['id'],
                TaskStatusUpdateEvent(
                    task_id=task['id'],
                    context_id=task['context_id'],
                    kind='status-update',
                    status={'state': 'failed'},
                    final=True,
                ),
            )
            raise
        else:
            await self.storage.update_task(
                task['id'], state='completed', new_artifacts=artifacts, new_messages=a2a_messages
            )

            # Send completion status streaming event
            await self.broker.send_stream_event(
                task['id'],
                TaskStatusUpdateEvent(
                    task_id=task['id'],
                    context_id=task['context_id'],
                    kind='status-update',
                    status={'state': 'completed'},
                    final=True,
                ),
            )

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
        return [Artifact(artifact_id=artifact_id, name='result', parts=[part])]

    def _convert_result_to_part(self, result: WorkerOutputT) -> Part:
        """Convert agent result to a Part (TextPart or DataPart).

        For string results, returns a TextPart.
        For structured data, returns a DataPart with properly serialized data.
        """
        if isinstance(result, str):
            return A2ATextPart(kind='text', text=result)
        else:
            output_type = type(result)
            type_adapter = TypeAdapter(output_type)
            data = type_adapter.dump_python(result, mode='json')
            json_schema = type_adapter.json_schema(mode='serialization')
            return DataPart(kind='data', data={'result': data}, metadata={'json_schema': json_schema})

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
                if 'bytes' in file_content:
                    data = file_content['bytes'].encode('utf-8')
                    mime_type = file_content.get('mime_type', 'application/octet-stream')
                    content = BinaryContent(data=data, media_type=mime_type)
                    model_parts.append(UserPromptPart(content=[content]))
                else:
                    url = file_content['uri']
                    for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
                        content = url_cls(url=url)
                        try:
                            content.media_type
                        except ValueError:  # pragma: no cover
                            continue
                        else:
                            break
                    else:
                        raise ValueError(f'Unsupported file type: {url}')  # pragma: no cover
                    model_parts.append(UserPromptPart(content=[content]))
            elif part['kind'] == 'data':
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
                a2a_parts.append(A2ATextPart(kind='text', text=part.content))
            elif isinstance(part, ThinkingPart):
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
