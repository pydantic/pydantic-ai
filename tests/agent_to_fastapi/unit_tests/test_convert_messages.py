# oragle/agent_libraries/oragle_agent/oragle_agent/tests/unit_tests/test_convert_messages.py

import json
from types import SimpleNamespace
from typing import Any

import pytest
from freezegun import freeze_time

from ...conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import (
        ChatCompletionMessageParam,
    )
    from openai.types.responses import (
        ResponseInputParam,
    )

    from pydantic_ai.fastapi.convert import (
        openai_chat_completions_2pai,
        openai_responses_input_to_pai,
        pai_result_to_openai_completions,
        pai_result_to_openai_responses,
    )


from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed or FastAPI not installed'),
    pytest.mark.anyio,
]


@pytest.fixture
def chat_user_with_image() -> list[dict[str, Any]]:
    """Chat-style user message with a text and an image_url part."""
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'see this image'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.com/img.png'}},
            ],
        },
    ]


@pytest.fixture
def chat_messages_with_tool_calls() -> list[dict[str, Any]]:
    """Messages exercising an assistant function/tool call flow."""
    return [
        {'role': 'system', 'content': 'system prompt'},
        {'role': 'user', 'content': 'please call tool'},
        {
            'role': 'assistant',
            'content': 'assistant invoking tool',
            'tool_calls': [
                {
                    'type': 'function',
                    'id': 'call_1',
                    'function': {'name': 'myfunc', 'arguments': {'x': 1}},
                },
            ],
        },
        {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'tool output', 'name': 'myfunc'},
        {'role': 'function', 'name': 'myfunc', 'content': 'func return content'},
    ]


@pytest.fixture
def responses_assistant_content() -> list[dict[str, Any]]:
    """Responses-style assistant content with output_text and a refusal."""
    return [
        {'type': 'message', 'role': 'user', 'content': 'user asks'},
        {
            'type': 'message',
            'role': 'assistant',
            'content': [
                {'type': 'output_text', 'text': 'hello world'},
                {'type': 'refusal', 'refusal': 'I refuse'},
            ],
        },
    ]


@pytest.fixture
def responses_function_call_and_output() -> list[dict[str, Any]]:
    """Function_call followed by its function_call_output (Responses format)."""
    return [
        {'type': 'message', 'role': 'user', 'content': 'start'},
        {'type': 'function_call', 'name': 'toolA', 'call_id': 'c1', 'arguments': {'a': 1}},
        {
            'type': 'function_call_output',
            'call_id': 'c1',
            'output': [{'type': 'output_text', 'text': 'resultA'}],
        },
    ]


@pytest.fixture
def responses_builtin_calls() -> list[dict[str, Any]]:
    """Examples of built-in Responses API tool calls."""
    return [
        {'type': 'message', 'role': 'user', 'content': 'begin'},
        {'type': 'file_search_call', 'id': 'f1', 'queries': ['a', 'b'], 'results': [{'name': 'x'}]},
        {'type': 'image_generation_call', 'id': 'img1', 'result': [{'url': 'https://img'}]},
        {
            'type': 'code_interpreter_call',
            'id': 'ci1',
            'container_id': 'cont',
            'outputs': [{'path': '/out'}],
            'code': 'print(1)',
        },
    ]


class MinimalAgentRunResult(AgentRunResult[str]):
    """Minimal AgentRunResult for tests: only `output` and `all_messages` are required."""

    def __init__(self, output: str, messages: list[Any]) -> None:
        self.output = output
        self._messages = messages

    def all_messages(
        self,
        *,
        output_tool_return_content: str | None = None,
    ) -> list[Any]:
        """Return the message history for the run."""
        return self._messages


@pytest.fixture
def fake_agent_result() -> AgentRunResult[str]:
    """Minimal AgentRunResult instance with `output` and `all_messages()`."""
    fake_msg = SimpleNamespace(provider_response_id=None, timestamp=None)
    return MinimalAgentRunResult(output='agent output text', messages=[fake_msg])


# --- Helpers ------------------------------------------------------------------


def collect_parts_of_type(
    messages: list[ModelMessage],
    part_type: type[ModelRequestPart | ModelResponsePart],
) -> list[ModelRequestPart | ModelResponsePart]:
    """Collect all parts of the given type from messages."""
    found: list[ModelRequestPart | ModelResponsePart] = []
    for message in messages:
        for part in getattr(message, 'parts', []):
            if isinstance(part, part_type):
                found.append(part)
    return found


# --- Tests -------------------------------------------------------------------


@pytest.mark.parametrize(
    ('input_value', 'expected_content'),
    [
        ('plain text input', 'plain text input'),
        ([{'role': 'user', 'content': 'plain user list'}], 'plain user list'),
    ],
)
@freeze_time('1970-01-01')
def test_openai_chat_completions_2pai_basic_string_and_list(
    input_value: list[ChatCompletionMessageParam] | str,
    expected_content: str,
) -> None:
    """ChatCompletions converter produces a ModelRequest with a UserPromptPart."""
    parsed = openai_chat_completions_2pai(input_value)
    assert parsed, 'Expected non-empty result'

    expected = [ModelRequest(parts=[UserPromptPart(content=expected_content)])]
    assert parsed == expected


@freeze_time('1970-01-01')
def test_openai_chat_completions_2pai_user_with_image(
    chat_user_with_image: list[ChatCompletionMessageParam] | str,
) -> None:
    """image_url parts become ImageUrl instances and preserve the URL."""
    parsed = openai_chat_completions_2pai(chat_user_with_image)
    assert parsed, 'Expected parsed messages'

    expected = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'see this image',
                        ImageUrl(url='https://example.com/img.png'),
                    ],
                ),
            ],
        ),
    ]
    assert parsed == expected


@freeze_time('1970-01-01')
def test_openai_chat_completions_2pai_assistant_tool_flow(
    chat_messages_with_tool_calls: list[ChatCompletionMessageParam] | str,
) -> None:
    """Tool calls and returns are converted into ToolCallPart and ToolReturnPart entries."""
    parsed = openai_chat_completions_2pai(chat_messages_with_tool_calls)
    expected = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='system prompt'),
                UserPromptPart(content='please call tool'),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(content='assistant invoking tool'),
                ToolCallPart(tool_name='myfunc', args={'x': 1}, tool_call_id='call_1'),
            ],
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='myfunc', content='tool output', tool_call_id='call_1'),
                ToolReturnPart(
                    tool_name='myfunc',
                    content='func return content',
                    tool_call_id='call_myfunc',
                ),
            ],
        ),
    ]
    assert parsed == expected


@pytest.mark.parametrize(
    ('items', 'expected_text'),
    [
        ('just a string', 'just a string'),
        ([{'type': 'message', 'role': 'user', 'content': 'some iterable'}], 'some iterable'),
    ],
)
@freeze_time('1970-01-01')
def test_openai_responses_input_to_pai_string_variants(
    items: ResponseInputParam | str,
    expected_text: str,
) -> None:
    """Responses-style strings and message lists produce a UserPromptPart."""
    parsed = openai_responses_input_to_pai(items)
    assert parsed, 'Expected parsed messages'

    expected = [ModelRequest(parts=[UserPromptPart(content=expected_text)])]
    assert parsed == expected


@freeze_time('1970-01-01')
def test_openai_responses_input_to_pai_assistant_content(
    responses_assistant_content: ResponseInputParam | str,
) -> None:
    """output_text -> TextPart; refusal -> prefixed text."""
    parsed = openai_responses_input_to_pai(responses_assistant_content)
    expected = [
        ModelRequest(parts=[UserPromptPart(content='user asks')]),
        ModelResponse(
            parts=[
                TextPart(content='hello world'),
                TextPart(content='[REFUSAL] I refuse'),
            ],
        ),
    ]
    assert parsed == expected


@freeze_time('1970-01-01')
def test_openai_responses_input_to_pai_function_call_and_output(
    responses_function_call_and_output: ResponseInputParam | str,
) -> None:
    """function_call + function_call_output -> ToolCallPart and ToolReturnPart."""
    parsed = openai_responses_input_to_pai(responses_function_call_and_output)

    expected = [
        ModelRequest(parts=[UserPromptPart(content='start')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='toolA', args={'a': 1}, tool_call_id='c1'),
            ],
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='toolA', content='resultA', tool_call_id='c1'),
            ],
        ),
    ]

    assert parsed == expected


@freeze_time('1970-01-01')
def test_openai_responses_input_to_pai_builtin_calls(
    responses_builtin_calls: ResponseInputParam,
) -> None:
    """Built-in calls (file_search, image_generation, code_interpreter) produce BuiltinToolCall/Return parts."""
    parsed = openai_responses_input_to_pai(responses_builtin_calls)

    expected = [
        ModelRequest(parts=[UserPromptPart(content='begin')]),
        ModelResponse(
            parts=[
                BuiltinToolCallPart(
                    tool_name='file_search',
                    args=json.dumps({'queries': ['a', 'b']}),
                    tool_call_id='f1',
                    provider_name='openai',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                BuiltinToolReturnPart(
                    tool_name='file_search',
                    content=[{'name': 'x'}],
                    tool_call_id='f1',
                    provider_name='openai',
                ),
                BuiltinToolCallPart(
                    tool_name='image_generation',
                    args=None,
                    tool_call_id='img1',
                    provider_name='openai',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                BuiltinToolReturnPart(
                    tool_name='image_generation',
                    content=[{'url': 'https://img'}],
                    tool_call_id='img1',
                    provider_name='openai',
                ),
                BuiltinToolCallPart(
                    tool_name='code_interpreter',
                    args=json.dumps({'code': 'print(1)', 'container_id': 'cont'}),
                    tool_call_id='ci1',
                    provider_name='openai',
                ),
            ],
        ),
        ModelResponse(
            parts=[
                BuiltinToolReturnPart(
                    tool_name='code_interpreter',
                    content=[{'path': '/out'}],
                    tool_call_id='ci1',
                    provider_name='openai',
                ),
            ],
        ),
    ]

    assert parsed == expected


@freeze_time('1970-01-01')
def test_pai_result_to_openai_completions_and_responses(fake_agent_result: AgentRunResult) -> None:
    """Convert AgentRunResult-like object into OpenAI ChatCompletion and Responses outputs."""
    expected_text = 'agent output text'

    chat = pai_result_to_openai_completions(fake_agent_result, model='unit-test-model')
    assert chat.model == 'unit-test-model'
    assert getattr(chat, 'choices', None), 'Expected at least one choice in ChatCompletion response'
    assert chat.choices[0].message.content == expected_text

    resp = pai_result_to_openai_responses(fake_agent_result, model='unit-test-model')
    assert resp.model == 'unit-test-model'
    assert getattr(resp, 'output', None), 'Expected Responses output to be non-empty'

    output_msg = resp.output[0]
    assert getattr(output_msg, 'role', None) == 'assistant'

    content_texts = [getattr(part, 'text', None) for part in output_msg.content]  # type:ignore
    assert expected_text in content_texts
