from typing import TypeGuard, TypeVar, Literal, Any, Callable

from openai.types.chat import CompletionCreateParams, ChatCompletionToolUnionParam, ChatCompletionFunctionToolParam, \
    ChatCompletionCustomToolParam, ChatCompletionContentPartParam, ChatCompletionContentPartTextParam, \
    ChatCompletionContentPartImageParam, ChatCompletionContentPartInputAudioParam, ChatCompletionMessageParam, \
    ChatCompletionAssistantMessageParam, ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionToolMessageParam, ChatCompletionFunctionMessageParam, ChatCompletionMessageToolCallUnionParam, \
    ChatCompletionMessageFunctionToolCallParam, ChatCompletionMessageCustomToolCallParam
from openai.types.chat.chat_completion_assistant_message_param import ContentArrayOfContentPart
from openai.types.chat.chat_completion_content_part_param import File
from openai.types.chat.chat_completion_content_part_refusal_param import ChatCompletionContentPartRefusalParam
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming, \
    CompletionCreateParamsNonStreaming


# Streaming params
def _is_streaming_params(run_input: CompletionCreateParams) -> TypeGuard[CompletionCreateParamsStreaming]:
    return isinstance(run_input, dict) and run_input.get("stream")


def _is_non_streaming_params(run_input: CompletionCreateParams) -> TypeGuard[CompletionCreateParamsNonStreaming]:
    return isinstance(run_input, dict) and not run_input.get("stream")


# Tool params
def _is_function_tool_param(param: ChatCompletionToolUnionParam) -> TypeGuard[ChatCompletionFunctionToolParam]:
    return isinstance(param, dict) and param.get('type') == 'function'


def _is_custom_tool_param(param: ChatCompletionToolUnionParam) -> TypeGuard[ChatCompletionCustomToolParam]:
    return isinstance(param, dict) and param.get('type') == 'custom'


# Content parts
def _is_text_part(part: ChatCompletionContentPartParam) -> TypeGuard[ChatCompletionContentPartTextParam]:
    return isinstance(part, dict) and part.get('type') == 'text'


def _is_image_part(part: ChatCompletionContentPartParam) -> TypeGuard[ChatCompletionContentPartImageParam]:
    return isinstance(part, dict) and part.get('type') == 'image_url'


def _is_audio_part(part: ChatCompletionContentPartParam) -> TypeGuard[ChatCompletionContentPartInputAudioParam]:
    return isinstance(part, dict) and part.get('type') == 'input_audio'


def _is_file_part(part: ChatCompletionContentPartParam) -> TypeGuard[File]:
    return isinstance(part, dict) and part.get('type') == 'file'


# Messages
def _is_assistant_message(message: ChatCompletionMessageParam) -> TypeGuard[ChatCompletionAssistantMessageParam]:
    return isinstance(message, dict) and message.get('role') == 'assistant'


def _is_system_message(message: ChatCompletionMessageParam) -> TypeGuard[ChatCompletionSystemMessageParam]:
    return isinstance(message, dict) and message.get('role') == 'system'


def _is_user_message(message: ChatCompletionMessageParam) -> TypeGuard[ChatCompletionUserMessageParam]:
    return isinstance(message, dict) and message.get('role') == 'user'


def _is_tool_message(message: ChatCompletionMessageParam) -> TypeGuard[ChatCompletionToolMessageParam]:
    return isinstance(message, dict) and message.get('role') == 'tool'


def _is_function_message(message: ChatCompletionMessageParam) -> TypeGuard[ChatCompletionFunctionMessageParam]:
    return isinstance(message, dict) and message.get('role') == 'function'


# Assistant content parts
def _is_assistant_text_part(part: ContentArrayOfContentPart) -> TypeGuard[ChatCompletionContentPartTextParam]:
    return isinstance(part, dict) and part.get('type') == 'text'


def _is_assistant_refusal_part(part: ContentArrayOfContentPart) -> TypeGuard[ChatCompletionContentPartRefusalParam]:
    return isinstance(part, dict) and part.get('type') == 'refusal'


# Tool call params
def _is_message_function_tool_call_param(
        part: ChatCompletionMessageToolCallUnionParam
) -> TypeGuard[ChatCompletionMessageFunctionToolCallParam]:
    return isinstance(part, dict) and part.get('type') == 'function'


def _is_message_custom_tool_call_param(
        part: ChatCompletionMessageToolCallUnionParam
) -> TypeGuard[ChatCompletionMessageCustomToolCallParam]:
    return isinstance(part, dict) and part.get('type') == 'custom'
