from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, assert_never

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

from .schema import Message, Part, TaskSendParams, TextPart as A2ATextPart

if TYPE_CHECKING:
    from pydantic_ai.agent import Agent

    from .worker import TaskContext


class Runner(ABC):
    """A runner is responsible for executing tasks."""

    @abstractmethod
    async def run(self, task_ctx: TaskContext[TaskSendParams]) -> None: ...

    @abstractmethod
    def build_message_history(self, task_history: list[Message]) -> list[ModelMessage]: ...

    @abstractmethod
    def build_agent_message(self, result: Any) -> Message: ...


@dataclass
class AgentRunner(Runner):
    """A runner that uses an agent to execute tasks."""

    agent: Agent

    async def run(self, task_ctx: TaskContext[TaskSendParams]) -> None:
        params = task_ctx.params
        task = await task_ctx.storage.load_task(params['id'], history_length=params.get('history_length'))
        assert task is not None, f'Task {task_ctx.params["id"]} not found'
        assert 'session_id' in task, 'Task must have a session_id'

        task_history = task.get('history', [])
        task_history.append(params['message'])
        message_history = self.build_message_history(task_history=task_history)

        # TODO(Marcelo): We need to make this more customizable.
        result = await self.agent.run(message_history=message_history)
        agent_message = self.build_agent_message(result.output)

        # TODO(Marcelo): This is in case of image/video/audio/file - which we don't support on the agent yet.
        # artifacts = self.build_artifacts(result.output)

        await task_ctx.storage.complete_task(task['id'], message=agent_message)

    def build_agent_message(self, result: Any) -> Message:
        # TODO(Marcelo): We need to send the json schema of the result on the metadata of the message.
        return Message(role='agent', parts=[A2ATextPart(type='text', text=str(result))])

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
