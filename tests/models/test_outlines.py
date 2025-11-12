# There are linting escapes for vllm offline as the CI would not contain the right
# environment to load the associated dependencies

# pyright: reportUnnecessaryTypeIgnoreComment = false

from __future__ import annotations as _annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    FilePart,
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
from pydantic_ai.settings import ModelSettings

from ..conftest import IsBytes, IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    import outlines

    from pydantic_ai.models.outlines import OutlinesModel
    from pydantic_ai.providers.outlines import OutlinesProvider

with try_import() as transformer_imports_successful:
    import transformers

with try_import() as llama_cpp_imports_successful:
    import llama_cpp

with try_import() as vllm_imports_successful:
    import vllm  # type: ignore[reportMissingImports]

    # We try to load the vllm model to ensure it is available
    try:  # pragma: no lax cover
        vllm.LLM('microsoft/Phi-3-mini-4k-instruct')  # type: ignore
    except RuntimeError as e:  # pragma: lax no cover
        if 'Found no NVIDIA driver' in str(e) or 'Device string must not be empty' in str(e):
            # Treat as import failure
            raise ImportError('CUDA/NVIDIA driver not available') from e
        raise

with try_import() as sglang_imports_successful:
    import openai

with try_import() as mlxlm_imports_successful:
    import mlx_lm


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='outlines not installed'),
    pytest.mark.anyio,
]

skip_if_transformers_imports_unsuccessful = pytest.mark.skipif(
    not transformer_imports_successful(), reason='transformers not available'
)

skip_if_llama_cpp_imports_unsuccessful = pytest.mark.skipif(
    not llama_cpp_imports_successful(), reason='llama_cpp not available'
)

skip_if_vllm_imports_unsuccessful = pytest.mark.skipif(not vllm_imports_successful(), reason='vllm not available')

skip_if_sglang_imports_unsuccessful = pytest.mark.skipif(not sglang_imports_successful(), reason='openai not available')

skip_if_mlxlm_imports_unsuccessful = pytest.mark.skipif(not mlxlm_imports_successful(), reason='mlx_lm not available')


@pytest.fixture
def mock_async_model() -> OutlinesModel:
    class MockOutlinesAsyncModel(outlines.models.base.AsyncModel):
        """Mock an OutlinesAsyncModel because no Outlines local models have an async version.

        The `__call__` and `stream` methods will be called by the Pydantic AI model while the other methods are
        only implemented because they are abstract methods in the OutlinesAsyncModel class.
        """

        async def __call__(self, model_input, output_type, backend, **inference_kwargs):  # type: ignore[reportMissingParameterType]
            return 'test'

        async def stream(self, model_input, output_type, backend, **inference_kwargs):  # type: ignore[reportMissingParameterType]
            for _ in range(2):
                yield 'test'

        async def generate(self, model_input, output_type, **inference_kwargs):  # type: ignore[reportMissingParameterType]
            ...  # pragma: no cover

        async def generate_batch(self, model_input, output_type, **inference_kwargs):  # type: ignore[reportMissingParameterType]
            ...  # pragma: no cover

        async def generate_stream(self, model_input, output_type, **inference_kwargs):  # type: ignore[reportMissingParameterType]
            ...  # pragma: no cover

    return OutlinesModel(MockOutlinesAsyncModel(), provider=OutlinesProvider())


@pytest.fixture
def transformers_model() -> OutlinesModel:
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore
        'erwanf/gpt2-mini',
        device_map='cpu',
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('erwanf/gpt2-mini')  # type: ignore
    chat_template = '{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}'
    hf_tokenizer.chat_template = chat_template
    outlines_model = outlines.models.transformers.from_transformers(
        hf_model,  # type: ignore[reportUnknownArgumentType]
        hf_tokenizer,  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def transformers_multimodal_model() -> OutlinesModel:
    hf_model = transformers.LlavaForConditionalGeneration.from_pretrained(  # type: ignore
        'trl-internal-testing/tiny-LlavaForConditionalGeneration',
        device_map='cpu',
    )
    hf_processor = transformers.AutoProcessor.from_pretrained(  # type: ignore
        'trl-internal-testing/tiny-LlavaForConditionalGeneration'
    )
    outlines_model = outlines.models.transformers.from_transformers(
        hf_model,
        hf_processor,  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def llamacpp_model() -> OutlinesModel:
    outlines_model_llamacpp = outlines.models.llamacpp.from_llamacpp(
        llama_cpp.Llama.from_pretrained(  # type: ignore
            repo_id='M4-ai/TinyMistral-248M-v2-Instruct-GGUF',
            filename='TinyMistral-248M-v2-Instruct.Q4_K_M.gguf',
        )
    )
    return OutlinesModel(outlines_model_llamacpp, provider=OutlinesProvider())


@pytest.fixture
def mlxlm_model() -> OutlinesModel:  # pragma: no cover
    outlines_model = outlines.models.mlxlm.from_mlxlm(
        *mlx_lm.load('mlx-community/SmolLM-135M-Instruct-4bit')  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def sglang_model() -> OutlinesModel:
    outlines_model = outlines.models.sglang.from_sglang(
        openai.OpenAI(api_key='test'),
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def vllm_model_offline() -> OutlinesModel:  # pragma: no cover
    outlines_model = outlines.models.vllm_offline.from_vllm_offline(  # type: ignore
        vllm.LLM('microsoft/Phi-3-mini-4k-instruct')  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def binary_image() -> BinaryImage:
    image_path = Path(__file__).parent.parent / 'assets' / 'kiwi.png'
    image_bytes = image_path.read_bytes()
    return BinaryImage(data=image_bytes, media_type='image/png')


outlines_parameters = [
    pytest.param(
        'from_transformers',
        lambda: (
            transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore
                'erwanf/gpt2-mini',
                device_map='cpu',
            ),
            transformers.AutoTokenizer.from_pretrained('erwanf/gpt2-mini'),  # type: ignore
        ),
        marks=skip_if_transformers_imports_unsuccessful,
    ),
    pytest.param(
        'from_llamacpp',
        lambda: (
            llama_cpp.Llama.from_pretrained(  # type: ignore
                repo_id='M4-ai/TinyMistral-248M-v2-Instruct-GGUF',
                filename='TinyMistral-248M-v2-Instruct.Q4_K_M.gguf',
            ),
        ),
        marks=skip_if_llama_cpp_imports_unsuccessful,
    ),
    pytest.param(
        'from_mlxlm',
        lambda: mlx_lm.load('mlx-community/SmolLM-135M-Instruct-4bit'),  # type: ignore
        marks=skip_if_mlxlm_imports_unsuccessful,
    ),
    pytest.param(
        'from_sglang',
        lambda: (openai.OpenAI(api_key='test'),),
        marks=skip_if_sglang_imports_unsuccessful,
    ),
    pytest.param(
        'from_vllm_offline',
        lambda: (vllm.LLM('microsoft/Phi-3-mini-4k-instruct'),),  # type: ignore
        marks=skip_if_vllm_imports_unsuccessful,
    ),
]


@pytest.mark.parametrize('model_loading_function_name,args', outlines_parameters)
def test_init(model_loading_function_name: str, args: Callable[[], tuple[Any]]) -> None:
    outlines_loading_function = getattr(outlines.models, model_loading_function_name)
    outlines_model = outlines_loading_function(*args())
    m = OutlinesModel(outlines_model, provider=OutlinesProvider())
    assert isinstance(m.model, outlines.models.base.Model | outlines.models.base.AsyncModel)
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


pydantic_ai_parameters = [
    pytest.param(
        'from_transformers',
        lambda: (
            transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore
                'erwanf/gpt2-mini',
                device_map='cpu',
            ),
            transformers.AutoTokenizer.from_pretrained('erwanf/gpt2-mini'),  # type: ignore
        ),
        marks=skip_if_transformers_imports_unsuccessful,
    ),
    pytest.param(
        'from_llamacpp',
        lambda: (
            llama_cpp.Llama.from_pretrained(  # type: ignore
                repo_id='M4-ai/TinyMistral-248M-v2-Instruct-GGUF',
                filename='TinyMistral-248M-v2-Instruct.Q4_K_M.gguf',
            ),
        ),
        marks=skip_if_llama_cpp_imports_unsuccessful,
    ),
    pytest.param(
        'from_mlxlm',
        lambda: mlx_lm.load('mlx-community/SmolLM-135M-Instruct-4bit'),  # type: ignore
        marks=skip_if_mlxlm_imports_unsuccessful,
    ),
    pytest.param(
        'from_sglang',
        lambda: ('https://example.com/', 'test'),
        marks=skip_if_sglang_imports_unsuccessful,
    ),
    pytest.param(
        'from_vllm_offline',
        lambda: (vllm.LLM('microsoft/Phi-3-mini-4k-instruct'),),  # type: ignore
        marks=skip_if_vllm_imports_unsuccessful,
    ),
]


@pytest.mark.parametrize('model_loading_function_name,args', pydantic_ai_parameters)
def test_model_loading_methods(model_loading_function_name: str, args: Callable[[], tuple[Any]]) -> None:
    loading_method = getattr(OutlinesModel, model_loading_function_name)
    m = loading_method(*args(), provider=OutlinesProvider())
    assert isinstance(m.model, outlines.models.base.Model | outlines.models.base.AsyncModel)
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


@skip_if_llama_cpp_imports_unsuccessful
async def test_request_async(llamacpp_model: OutlinesModel) -> None:
    agent = Agent(llamacpp_model, instructions='Answer in one word.')
    result = await agent.run('What is the capital of France?', model_settings=ModelSettings(max_tokens=100))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Answer in one word.',
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )
    result = await agent.run('What is the capital of Germany?', message_history=result.all_messages())
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Answer in one word.',
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Germany?',
                        timestamp=IsDatetime(),
                    )
                ],
                instructions='Answer in one word.',
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )


@skip_if_llama_cpp_imports_unsuccessful
def test_request_sync(llamacpp_model: OutlinesModel) -> None:
    agent = Agent(llamacpp_model)
    result = agent.run_sync('What is the capital of France?', model_settings=ModelSettings(max_tokens=100))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )


@skip_if_llama_cpp_imports_unsuccessful
async def test_request_streaming(llamacpp_model: OutlinesModel) -> None:
    agent = Agent(llamacpp_model)
    async with agent.run_stream(
        'What is the capital of the UK?', model_settings=ModelSettings(max_tokens=100)
    ) as response:
        async for text in response.stream_text():
            assert isinstance(text, str)
            assert len(text) > 0


async def test_request_async_model(mock_async_model: OutlinesModel) -> None:
    agent = Agent(mock_async_model)
    result = await agent.run('What is the capital of France?', model_settings=ModelSettings(max_tokens=100))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )


async def test_request_streaming_async_model(mock_async_model: OutlinesModel) -> None:
    agent = Agent(mock_async_model)
    async with agent.run_stream(
        'What is the capital of the UK?', model_settings=ModelSettings(max_tokens=100)
    ) as response:
        async for text in response.stream_text():
            assert isinstance(text, str)
            assert len(text) > 0


@skip_if_transformers_imports_unsuccessful
def test_request_image_binary(transformers_multimodal_model: OutlinesModel, binary_image: BinaryImage) -> None:
    agent = Agent(transformers_multimodal_model)
    result = agent.run_sync(
        ["What's on the image?", binary_image], model_settings=ModelSettings(extra_body={'max_new_tokens': 100})
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What's on the image?",
                            BinaryImage(data=IsBytes(), media_type='image/png'),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )


@skip_if_transformers_imports_unsuccessful
def test_request_image_url(transformers_multimodal_model: OutlinesModel) -> None:
    agent = Agent(transformers_multimodal_model)
    result = agent.run_sync(
        [
            "What's on the image?",
            ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'),
        ],
        model_settings=ModelSettings(extra_body={'max_new_tokens': 100}),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            "What's on the image?",
                            ImageUrl(
                                url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg'
                            ),
                        ],
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )


@skip_if_llama_cpp_imports_unsuccessful
def test_tool_definition(llamacpp_model: OutlinesModel) -> None:
    # function tools
    agent = Agent(llamacpp_model, builtin_tools=[WebSearchTool()])
    with pytest.raises(UserError, match='Outlines does not support function tools and builtin tools yet.'):
        agent.run_sync('Hello')

    # built-in tools
    agent = Agent(llamacpp_model)

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

    agent = Agent(llamacpp_model, output_type=ToolOutput(MyOutput, name='my_output_tool'))
    with pytest.raises(UserError, match='Tool output is not supported by this model.'):
        agent.run_sync('Hello')


@skip_if_llama_cpp_imports_unsuccessful
def test_output_type(llamacpp_model: OutlinesModel) -> None:
    class Box(BaseModel):
        width: int
        height: int
        depth: int
        units: int

    agent = Agent(llamacpp_model, output_type=Box)
    result = agent.run_sync('Give me the dimensions of a box', model_settings=ModelSettings(max_tokens=100))
    assert isinstance(result.output, Box)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Give me the dimensions of a box',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(parts=[TextPart(content=IsStr())], timestamp=IsDatetime(), run_id=IsStr()),
        ]
    )


@skip_if_transformers_imports_unsuccessful
def test_input_format(transformers_multimodal_model: OutlinesModel, binary_image: BinaryImage) -> None:
    agent = Agent(transformers_multimodal_model)

    # all accepted message types
    message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a helpful assistance'),
                UserPromptPart(content='Hello'),
                RetryPromptPart(content='Failure'),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart('Thinking...'),  # ignored by the model
                TextPart('Hello there!'),
                FilePart(content=binary_image),
            ]
        ),
    ]
    agent.run_sync('How are you doing?', message_history=message_history)

    # unsupported: non-image multi-modal user prompts
    multi_modal_message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'Hello there!',
                        AudioUrl('https://example.com/audio.mp3'),
                    ]
                )
            ]
        )
    ]
    with pytest.raises(
        UserError, match='Each element of the content sequence must be a string, an `ImageUrl` or a `BinaryImage`.'
    ):
        agent.run_sync('How are you doing?', message_history=multi_modal_message_history)

    # unsupported: tool calls
    tool_call_message_history: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart(tool_call_id='1', tool_name='get_location')]),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_location', content='London', tool_call_id='1')]),
    ]
    with pytest.raises(UserError, match='Tool calls are not supported for Outlines models yet.'):
        agent.run_sync('How are you doing?', message_history=tool_call_message_history)

    # unsupported: tool returns
    tool_return_message_history: list[ModelMessage] = [
        ModelRequest(parts=[ToolReturnPart(tool_name='get_location', content='London', tool_call_id='1')])
    ]
    with pytest.raises(UserError, match='Tool calls are not supported for Outlines models yet.'):
        agent.run_sync('How are you doing?', message_history=tool_return_message_history)

    # unsupported: non-image file parts
    file_part_message_history: list[ModelMessage] = [
        ModelResponse(parts=[FilePart(content=BinaryContent(data=b'test', media_type='text/plain'))])
    ]
    with pytest.raises(
        UserError, match='File parts other than `BinaryImage` are not supported for Outlines models yet.'
    ):
        agent.run_sync('How are you doing?', message_history=file_part_message_history)


@skip_if_transformers_imports_unsuccessful
def test_model_settings_transformers(transformers_model: OutlinesModel) -> None:
    # unsupported arguments removed
    kwargs = transformers_model.format_inference_kwargs(
        ModelSettings(
            timeout=1,
            parallel_tool_calls=True,
            seed=123,
            presence_penalty=0.5,
            frequency_penalty=0.5,
            stop_sequences=['.'],
            extra_headers={'Authorization': 'Bearer 123'},
        )
    )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'seed' not in kwargs
    assert 'presence_penalty' not in kwargs
    assert 'frequency_penalty' not in kwargs
    assert 'stop_sequences' not in kwargs
    assert 'extra_headers' not in kwargs

    # extra_body merging
    kwargs = transformers_model.format_inference_kwargs(
        ModelSettings(
            extra_body={'tokenizer': 'test_tokenizer'},
            max_tokens=100,
        )
    )
    assert kwargs['tokenizer'] == 'test_tokenizer'
    assert kwargs['max_tokens'] == 100
    assert 'extra_body' not in kwargs


@skip_if_llama_cpp_imports_unsuccessful
def test_model_settings_llamacpp(llamacpp_model: OutlinesModel) -> None:
    # unsupported arguments removed
    kwargs = llamacpp_model.format_inference_kwargs(
        ModelSettings(
            timeout=1,
            parallel_tool_calls=True,
            stop_sequences=['.'],
            extra_headers={'Authorization': 'Bearer 123'},
        )
    )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'stop_sequences' not in kwargs
    assert 'extra_headers' not in kwargs

    # extra_body merging
    kwargs = llamacpp_model.format_inference_kwargs(
        ModelSettings(
            extra_body={'min_p': 0.1},
            max_tokens=100,
        )
    )
    assert kwargs['min_p'] == 0.1
    assert kwargs['max_tokens'] == 100
    assert 'extra_body' not in kwargs


@skip_if_mlxlm_imports_unsuccessful
def test_model_settings_mlxlm(mlxlm_model: OutlinesModel) -> None:  # pragma: no cover
    # all arguments are removed
    kwargs = mlxlm_model.format_inference_kwargs(
        ModelSettings(
            temperature=0.7,
            top_p=0.9,
            timeout=1,
            parallel_tool_calls=True,
            seed=123,
            presence_penalty=0.7,
            frequency_penalty=0.3,
            logit_bias={'20': 5},
            stop_sequences=['Paris'],
            extra_headers={'Authorization': 'Bearer 123'},
        )
    )
    for setting in [
        'temperature',
        'top_p',
        'timeout',
        'parallel_tool_calls',
        'seed',
        'presence_penalty',
        'frequency_penalty',
        'logit_bias',
        'stop_sequences',
        'extra_headers',
    ]:
        assert setting not in kwargs

    # extra_body merging
    kwargs = mlxlm_model.format_inference_kwargs(
        ModelSettings(
            extra_body={'verbose': True},
        )
    )
    assert kwargs['verbose']
    assert 'extra_body' not in kwargs


@skip_if_sglang_imports_unsuccessful
def test_model_settings_sglang(sglang_model: OutlinesModel) -> None:
    # unsupported arguments removed
    kwargs = sglang_model.format_inference_kwargs(
        ModelSettings(
            timeout=1,
            parallel_tool_calls=True,
            seed=123,
            logit_bias={'20': 10},
            stop_sequences=['.'],
            extra_headers={'Authorization': 'Bearer 123'},
        )
    )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'seed' not in kwargs
    assert 'logit_bias' not in kwargs
    assert 'stop_sequences' not in kwargs
    assert 'extra_headers' not in kwargs

    # extra_body merging
    kwargs = sglang_model.format_inference_kwargs(
        ModelSettings(
            extra_body={'stop': ['.']},
            max_tokens=100,
        )
    )
    assert kwargs['stop'] == ['.']
    assert kwargs['max_tokens'] == 100
    assert 'extra_body' not in kwargs


@skip_if_vllm_imports_unsuccessful
def test_model_settings_vllm_offline(vllm_model_offline: OutlinesModel) -> None:  # pragma: no cover
    # unsupported arguments removed
    kwargs = vllm_model_offline.format_inference_kwargs(
        ModelSettings(
            timeout=1,
            parallel_tool_calls=True,
            stop_sequences=['.'],
            extra_headers={'Authorization': 'Bearer 123'},
        )
    )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'stop_sequences' not in kwargs
    assert 'extra_headers' not in kwargs

    # special keys are preserved and others are in sampling params
    kwargs = vllm_model_offline.format_inference_kwargs(
        ModelSettings(  # type: ignore[reportCallIssue]
            use_tqdm=True,
            lora_request='test',
            priority=1,
            temperature=1,
        )
    )
    assert kwargs['use_tqdm'] is True
    assert kwargs['lora_request'] == 'test'
    assert kwargs['priority'] == 1
    assert 'sampling_params' in kwargs
    assert 'temperature' in kwargs['sampling_params']
