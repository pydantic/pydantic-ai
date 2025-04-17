from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..agent import Agent
from .schema import Task, TaskSendParams


# TODO(Marcelo): Should the Runner be responsible for saving the artifact?
class Runner(ABC):
    """A runner is responsible for executing tasks."""

    @abstractmethod
    async def run(self, task_send_params: TaskSendParams, task: Task) -> None: ...


@dataclass
class AgentRunner(Runner):
    """A runner that uses an agent to execute tasks."""

    agent: Agent

    async def run(self, task_send_params: TaskSendParams, task: Task) -> None: ...

    # async def _handle_submitted_task(self, request: SendTaskRequest):
    #     task = request['params']
    #     history: list[Message] = task.get('history', [])
    #     result = await self.agent.run(message_history=self._map_history(history))
    #     task['status']['state'] = 'completed'
    #     # return result

    # def _map_history(self, history: list[Message]) -> list[ModelMessage]:
    #     model_messages: list[ModelMessage] = []
    #     for message in history:
    #         if message['role'] == 'user':
    #             # NOTE: This is the user input message.
    #             model_messages.append(ModelRequest(parts=self._map_request_parts(message['parts'])))
    #         else:
    #             model_messages.append(ModelResponse(parts=self._map_response_parts(message['parts'])))
    #     return model_messages

    # def _map_request_parts(self, parts: list[Part]) -> list[ModelRequestPart]:
    #     model_parts: list[ModelRequestPart] = []
    #     for part in parts:
    #         if part['type'] == 'text':
    #             model_parts.append(UserPromptPart(content=part['text']))
    #         elif part['type'] == 'file':
    #             file = part['file']
    #             if 'data' in file:
    #                 data = file['data'].encode('utf-8')
    #                 content = BinaryContent(data=data, media_type=file['mime_type'])
    #                 model_parts.append(UserPromptPart(content=[content]))
    #             else:
    #                 url = file['url']
    #                 for url_cls in (DocumentUrl, AudioUrl, ImageUrl, VideoUrl):
    #                     content = url_cls(url=url)
    #                     try:
    #                         content.media_type
    #                     except ValueError:
    #                         continue
    #                     else:
    #                         break
    #                 else:
    #                     raise ValueError(f'Unknown file type: {file["mime_type"]}')
    #                 model_parts.append(UserPromptPart(content=[content]))
    #         elif part['type'] == 'data':
    #             # TODO(Marcelo): Maybe we should use this for `ToolReturnPart`, and `RetryPromptPart`.
    #             raise NotImplementedError('Data parts are not supported yet.')
    #     return model_parts

    # def _map_response_parts(self, parts: list[Part]) -> list[ModelResponsePart]: ...
