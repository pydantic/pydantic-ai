from dataclasses import dataclass, field

from . import Model, ModelRequestParameters, ModelResponse, ModelMessage, check_allow_model_requests
from ..settings import ModelSettings
from ..result import Usage
from ..messages import TextPart, ModelResponsePart, ModelRequest
from ..tools import ToolDefinition

from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import (
    ChatCompletions,
    ChatCompletionsToolDefinition,
    FunctionDefinition,
    ChatRequestMessage,
    TextContentItem,
    UserMessage,
)

INFERENCE_ENDPOINT = "https://models.inference.ai.azure.com"

@dataclass(init=False)
class GitHubModels(Model):

    client: ChatCompletionsClient = field(repr=False)
    _model_name: str = field(default="gpt-4o", repr=False)
    _extra: dict[str, str] = field(default_factory=dict, repr=False)

    def __init__(self, model_name: str, *, api_key: str):
        self._model_name = model_name
        self._extra = {}
        if model_name == "o3-mini":
            self._extra["api_version"] = "2024-12-01-preview"

        self.client = ChatCompletionsClient(
            endpoint=INFERENCE_ENDPOINT,
            credential=AzureKeyCredential(api_key),
            model=model_name,
            **self._extra,
        )

    @staticmethod
    def _as_chat_request_message(messages: list[ModelMessage]) -> list[ChatRequestMessage]:
        result: list[ChatRequestMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                result.append(UserMessage(content=[TextContentItem(part.content) for part in message.parts if isinstance(part, str)]))
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")
        return result

    @staticmethod
    def _as_response_and_usage(response: ChatCompletions) -> tuple[ModelResponse, Usage]:
        usage = Usage(response_tokens=response.usage.completion_tokens, request_tokens=response.usage.prompt_tokens, total_tokens=response.usage.total_tokens)
        parts: list[ModelResponsePart] = [TextPart(content=choice.message.content) for choice in response.choices]
        model_response = ModelResponse(
            parts=parts,
        )
        return model_response, usage

    @staticmethod
    def _get_tool_choice(model_request_parameters: ModelRequestParameters) -> str | None:
        """Get tool choice for the model.

        - "auto": Default mode. Model decides if it uses the tool or not.
        - "none": Prevents tool use.
        - "required": Forces tool use.
        """
        if not model_request_parameters.function_tools and not model_request_parameters.result_tools:
            return None
        elif not model_request_parameters.allow_text_result:
            return 'required'
        else:
            return 'auto'

    @staticmethod
    def _as_tool_definition(model_request_parameters: ModelRequestParameters) -> list[ChatCompletionsToolDefinition] | None:
        """Map function and result tools to Inferencing SDK format.

        Returns None if both function_tools and result_tools are empty.
        """
        all_tools: list[ToolDefinition] = (
            model_request_parameters.function_tools + model_request_parameters.result_tools
        )
        tools = [
            ChatCompletionsToolDefinition(
                function=FunctionDefinition(name=r.name, parameters=r.parameters_json_schema, description=r.description)
            )
            for r in all_tools
        ]
        return tools if tools else None

    async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> tuple[ModelResponse, Usage]:
            """Make a non-streaming request to the model from Pydantic AI call."""
            check_allow_model_requests()
            model_settings = model_settings or {}
            response = await self.client.complete(
                messages=self._as_chat_request_message(messages),
                stream=False,
                model=self._model_name,
                tools=None, # TODO? 
                tool_choice=self._get_tool_choice(model_request_parameters),  # TODO?
                max_tokens=model_settings.get('max_tokens', None),
                temperature=model_settings.get('temperature', None),
                top_p=model_settings.get('top_p', 1),
                seed=model_settings.get('seed', None),
            )
            return self._as_response_and_usage(response)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "github"
