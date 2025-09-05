# We only test with the transformers model to limit the number of dependencies

import json
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.output import ToolOutput
from pydantic_ai.profiles import ModelProfile

from ..conftest import try_import

with try_import() as imports_successful:
    from outlines.models.transformers import Transformers, from_transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pydantic_ai.models.outlines import (
        OutlinesModel,
    )
    from pydantic_ai.providers.outlines import OutlinesProvider

if TYPE_CHECKING:
    from outlines.models.transformers import Transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='outlines not installed'),
    pytest.mark.anyio,
]


TRANSFORMERS_MODEL_NAME = 'erwanf/gpt2-mini'


@pytest.fixture
def outlines_model() -> Transformers:
    hf_model = AutoModelForCausalLM.from_pretrained(TRANSFORMERS_MODEL_NAME)  # type: ignore[no-untyped-call]
    hf_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_NAME)  # type: ignore[no-untyped-call]
    chat_template = '{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}'
    hf_tokenizer.chat_template = chat_template
    return from_transformers(model=hf_model, tokenizer_or_processor=hf_tokenizer)  # type: ignore[no-untyped-call]


def test_init(outlines_model: Any) -> None:
    m = OutlinesModel(outlines_model, provider=OutlinesProvider())
    assert isinstance(m.model, Transformers)
    assert m.model_name == 'outlines-model'
    assert m.system == 'outlines'
    assert m.settings is None
    assert m.profile == ModelProfile(
        supports_tools=False,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode='native',
        thinking_tags=('<think>', '</think>'),
        ignore_streamed_leading_whitespace=False,
    )


def test_from_transformers():
    m = OutlinesModel.from_transformers(
        AutoModelForCausalLM.from_pretrained(TRANSFORMERS_MODEL_NAME),  # type: ignore[no-untyped-call]
        AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_NAME),  # type: ignore[no-untyped-call]
    )
    assert isinstance(m.model, Transformers)
    assert m.model_name == 'outlines-model'
    assert m.system == 'outlines'
    assert m.settings is None
    assert m.profile == ModelProfile(
        supports_tools=False,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode='native',
        thinking_tags=('<think>', '</think>'),
        ignore_streamed_leading_whitespace=False,
    )


async def test_request_async(outlines_model: Transformers) -> None:
    m = OutlinesModel(outlines_model)
    agent = Agent(m)
    result = await agent.run('What is the capital of France?')
    assert len(result.output) > 0
    result = await agent.run('What is the capital of Germany?', message_history=result.all_messages())
    assert len(result.output) > 0
    all_messages = result.all_messages()
    assert len(all_messages) == 4

    assert isinstance(all_messages[0], ModelRequest)
    assert len(all_messages[0].parts) == 1
    assert isinstance(all_messages[0].parts[0], UserPromptPart)
    assert all_messages[0].parts[0].content == 'What is the capital of France?'

    assert isinstance(all_messages[1], ModelResponse)
    assert len(all_messages[1].parts) == 1
    assert isinstance(all_messages[1].parts[0], TextPart)
    assert isinstance(all_messages[1].parts[0].content, str)

    assert isinstance(all_messages[2], ModelRequest)
    assert len(all_messages[2].parts) == 1
    assert isinstance(all_messages[2].parts[0], UserPromptPart)
    assert all_messages[2].parts[0].content == 'What is the capital of Germany?'

    assert isinstance(all_messages[3], ModelResponse)
    assert len(all_messages[3].parts) == 1
    assert isinstance(all_messages[3].parts[0], TextPart)
    assert isinstance(all_messages[3].parts[0].content, str)


def test_request_sync(outlines_model: Transformers) -> None:
    m = OutlinesModel(outlines_model)
    agent = Agent(m)
    result = agent.run_sync('What is the capital of France?')
    assert len(result.output) > 0
    all_messages = result.all_messages()

    assert len(all_messages) == 2

    assert isinstance(all_messages[0], ModelRequest)
    assert len(all_messages[0].parts) == 1
    assert isinstance(all_messages[0].parts[0], UserPromptPart)
    assert all_messages[0].parts[0].content == 'What is the capital of France?'

    assert isinstance(all_messages[1], ModelResponse)
    assert len(all_messages[1].parts) == 1
    assert isinstance(all_messages[1].parts[0], TextPart)
    assert isinstance(all_messages[1].parts[0].content, str)


async def test_request_streaming(outlines_model: Transformers) -> None:
    # The transformers model does not support streaming,
    # so we need to mock the generate_stream method.
    def patched_generate_stream(self_ref: Transformers, *args: Any, **kwargs: Any) -> Any:
        response = self_ref.generate(*args, **kwargs)  # type: ignore[no-untyped-call]
        for i in range(0, len(response), 10):
            chunk = response[i : i + 10]
            yield chunk

    outlines_model.generate_stream = partial(patched_generate_stream, outlines_model)
    m = OutlinesModel(outlines_model)
    agent = Agent(m)
    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            assert isinstance(text, str)
            assert len(text) > 0


def test_tool_definition(outlines_model: Transformers) -> None:
    m = OutlinesModel(outlines_model)

    # function tools
    agent = Agent(m, builtin_tools=[WebSearchTool()])
    with pytest.raises(UserError, match='Outlines does not support function tools and builtin tools yet.'):
        agent.run_sync('Hello')

    # built-in tools
    agent = Agent(m)

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:  # pragma: no cover
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')

    with pytest.raises(UserError, match='Outlines does not support function tools and builtin tools yet.'):
        agent.run_sync('Hello')

    # output tools
    class MyOutput(BaseModel):
        name: str

    agent = Agent(m, output_type=ToolOutput(MyOutput, name='my_output_tool'))
    with pytest.raises(UserError, match='Output tools are not supported by the model.'):
        agent.run_sync('Hello')


def test_output_type(outlines_model: Transformers) -> None:
    class Box(BaseModel):
        width: int
        height: int
        depth: int
        units: int

    m = OutlinesModel(outlines_model)
    agent = Agent(m, output_type=Box)
    result = agent.run_sync('Give me the dimensions of a box', model_settings={'max_new_tokens': 100})  # type: ignore[typeddict-item]
    assert isinstance(result.output, Box)


def test_model_settings(outlines_model: Transformers) -> None:
    # set at model level + max_new_tokens
    model = OutlinesModel(outlines_model, settings={'max_new_tokens': 1})  # type: ignore[typeddict-item]
    agent = Agent(model)
    result = agent.run_sync('How are you doing?')
    assert len(result.output) < 10

    # set at agent level + stop_sequences
    model = OutlinesModel(outlines_model)
    agent = Agent(
        model,
        model_settings={  # type: ignore[typeddict-item]
            'stop_sequences': ['Paris'],
            'max_new_tokens': 200,
            'extra_body': {'tokenizer': outlines_model.hf_tokenizer},
        },
    )
    result = agent.run_sync('Write a story about Paris')
    assert result.output.endswith('Paris')

    # set at run level + args of ModelSettings that are not supported
    model = OutlinesModel(outlines_model)
    agent = Agent(model)
    with pytest.warns(UserWarning, match='The transformers model does not support'):
        result = agent.run_sync(
            'Hello',
            model_settings={
                'timeout': 1,
                'parallel_tool_calls': True,
                'seed': 123,
                'extra_headers': {'Authorization': 'Bearer 123'},
            },
        )
    assert isinstance(result.output, str)

    # presence_penalty and frequency_penalty
    with pytest.warns(UserWarning, match='The transformers model has a single argument `repetition_penalty`'):
        result = agent.run_sync('Hello', model_settings={'presence_penalty': 0.7, 'frequency_penalty': 0.3})
    assert isinstance(result.output, str)

    # logit_bias
    with pytest.warns(UserWarning, match='The transformers model expects the keys of the `logits_bias`'):
        result = agent.run_sync('Hello', model_settings={'logit_bias': {'20,21': 0.5, '22': 0.3, 'a': 0.2}})  # type: ignore[typeddict-item]
    assert isinstance(result.output, str)


def test_input_format(outlines_model: Transformers) -> None:
    m = OutlinesModel(outlines_model)
    agent = Agent(m)

    # all accepted message types
    message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistance'),
                UserPromptPart(content='Hello'),
                UserPromptPart(content=['Foo', 'Bar']),
                RetryPromptPart(content='Failure'),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart('Thinking...'),  # ignored by the model
                TextPart('Hello there!'),
            ]
        ),
    ]
    agent.run_sync('How are you doing?', message_history=message_history)

    # unsupported: multi-modal user prompts
    message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=['Describe the image', ImageUrl(url='https://example.com/image.png')])]
        )
    ]
    with pytest.raises(UserError, match='Outlines does not support multi-modal user prompts yet.'):
        agent.run_sync('How are you doing?', message_history=message_history)

    # unsupported: tool calls
    message_history: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_call_id='1', tool_name='get_location')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_location', content='London', tool_call_id='1')]),
    ]
    with pytest.raises(UserError, match='Tool calls are not supported for Outlines models yet.'):
        agent.run_sync('How are you doing?', message_history=message_history)

    # unsupported: tool returns
    message_history: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='get_location', content='London', tool_call_id='1')])
    ]
    with pytest.raises(UserError, match='Tool calls are not supported for Outlines models yet.'):
        agent.run_sync('How are you doing?', message_history=message_history)
