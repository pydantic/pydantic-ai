# There are linting escapes for vllm offline as the CI would not contain the right
# environment to load the associated dependencies

import json
from collections.abc import Callable
from typing import Any

import pytest
from inline_snapshot import snapshot
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

from ..conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    import outlines

    from pydantic_ai.models.outlines import (
        OutlinesModel,
        OutlinesModelSettings,
    )
    from pydantic_ai.providers.outlines import OutlinesProvider

with try_import() as transformer_imports_successful:
    import transformers

with try_import() as llama_cpp_imports_successful:
    import llama_cpp

with try_import() as vllm_imports_successful:
    import vllm  # type: ignore[reportMissingImports]

    # We try to load the vllm model to ensure it is available
    try:
        vllm.LLM('microsoft/Phi-3-mini-4k-instruct')  # type: ignore
    except RuntimeError as e:
        if 'Found no NVIDIA driver' in str(e):
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
def transformers_model() -> 'OutlinesModel':
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(  # type: ignore
        'erwanf/gpt2-mini',
        device_map='cpu',
    )
    hf_tokenizer = transformers.AutoTokenizer.from_pretrained('erwanf/gpt2-mini')  # type: ignore
    chat_template = '{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}'
    hf_tokenizer.chat_template = chat_template
    outlines_model = outlines.models.transformers.from_transformers(
        hf_model,  # type: ignore
        hf_tokenizer,  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def llamacpp_model() -> 'OutlinesModel':
    outlines_model_llamacpp = outlines.models.llamacpp.from_llamacpp(
        llama_cpp.Llama.from_pretrained(  # type: ignore
            repo_id='M4-ai/TinyMistral-248M-v2-Instruct-GGUF',
            filename='TinyMistral-248M-v2-Instruct.Q4_K_M.gguf',
        )
    )
    return OutlinesModel(outlines_model_llamacpp, provider=OutlinesProvider())


@pytest.fixture
def mlxlm_model() -> 'OutlinesModel':
    outlines_model = outlines.models.mlxlm.from_mlxlm(  # type: ignore
        *mlx_lm.load('mlx-community/SmolLM-135M-Instruct-4bit')  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def sglang_model() -> 'OutlinesModel':
    outlines_model = outlines.models.sglang.from_sglang(
        openai.OpenAI(api_key='test'),
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


@pytest.fixture
def vllm_model_offline() -> 'OutlinesModel':
    outlines_model = outlines.models.vllm_offline.from_vllm_offline(  # type: ignore
        vllm.LLM('microsoft/Phi-3-mini-4k-instruct')  # type: ignore
    )
    return OutlinesModel(outlines_model, provider=OutlinesProvider())


parameters = [
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


@pytest.mark.parametrize('model_loading_function_name,args', parameters)
def test_init(model_loading_function_name: str, args: Callable[[], tuple[Any]]) -> None:
    outlines_loading_function = getattr(outlines.models, model_loading_function_name)
    outlines_model = outlines_loading_function(*args())
    m = OutlinesModel(outlines_model, provider=OutlinesProvider())
    assert isinstance(m.model, outlines.models.Model)
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


@pytest.mark.parametrize('model_loading_function_name,args', parameters)
def test_model_loading_methods(model_loading_function_name: str, args: Callable[[], tuple[Any]]) -> None:
    loading_method = getattr(OutlinesModel, model_loading_function_name)
    m = loading_method(*args(), provider=OutlinesProvider())
    assert isinstance(m.model, outlines.models.Model)
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
async def test_request_async(llamacpp_model: 'OutlinesModel') -> None:
    agent = Agent(llamacpp_model)
    result = await agent.run('What is the capital of France?', model_settings=OutlinesModelSettings(max_tokens=100))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr())],
                timestamp=IsDatetime(),
            ),
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr())],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Germany?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr())],
                timestamp=IsDatetime(),
            ),
        ]
    )


@skip_if_llama_cpp_imports_unsuccessful
def test_request_sync(llamacpp_model: 'OutlinesModel') -> None:
    agent = Agent(llamacpp_model)
    result = agent.run_sync('What is the capital of France?', model_settings=OutlinesModelSettings(max_tokens=100))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content=IsStr())],
                timestamp=IsDatetime(),
            ),
        ]
    )


@skip_if_llama_cpp_imports_unsuccessful
async def test_request_streaming(llamacpp_model: 'OutlinesModel') -> None:
    agent = Agent(llamacpp_model)
    async with agent.run_stream(
        'What is the capital of the UK?', model_settings=OutlinesModelSettings(max_tokens=100)
    ) as response:
        async for text in response.stream_text():
            assert isinstance(text, str)
            assert len(text) > 0


@skip_if_llama_cpp_imports_unsuccessful
def test_tool_definition(llamacpp_model: 'OutlinesModel') -> None:
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
    with pytest.raises(UserError, match='Output tools are not supported by the model.'):
        agent.run_sync('Hello')


@skip_if_llama_cpp_imports_unsuccessful
def test_output_type(llamacpp_model: 'OutlinesModel') -> None:
    class Box(BaseModel):
        width: int
        height: int
        depth: int
        units: int

    agent = Agent(llamacpp_model, output_type=Box)
    result = agent.run_sync('Give me the dimensions of a box', model_settings=OutlinesModelSettings(max_tokens=100))
    assert isinstance(result.output, Box)


@skip_if_llama_cpp_imports_unsuccessful
def test_input_format(llamacpp_model: 'OutlinesModel') -> None:
    agent = Agent(llamacpp_model)

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
    multi_modal_message_history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart(content=['Describe the image', ImageUrl(url='https://example.com/image.png')])]
        )
    ]
    with pytest.raises(UserError, match='Outlines does not support multi-modal user prompts yet.'):
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


@skip_if_transformers_imports_unsuccessful
def test_model_settings_transformers(transformers_model: 'OutlinesModel') -> None:
    # unsupported arguments removed with warnings
    with pytest.warns(UserWarning, match='The transformers model does not support'):
        kwargs = transformers_model.format_inference_kwargs(
            OutlinesModelSettings(
                timeout=1,
                parallel_tool_calls=True,
                seed=123,
                extra_headers={'Authorization': 'Bearer 123'},
            )
        )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'seed' not in kwargs
    assert 'extra_headers' not in kwargs

    # repetition_penalty from presence_penalty and frequency_penalty
    with pytest.warns(UserWarning, match='The transformers model has a single argument `repetition_penalty`'):
        kwargs = transformers_model.format_inference_kwargs(
            OutlinesModelSettings(
                presence_penalty=0.7,
                frequency_penalty=0.3,
            )
        )
    assert kwargs['repetition_penalty'] == 0.3  # frequency_penalty takes precedence
    assert 'presence_penalty' not in kwargs
    assert 'frequency_penalty' not in kwargs

    # logit_bias: cast values and warn if not compatible
    with pytest.warns(UserWarning, match='The transformers model expects the keys of the `logits_bias`'):
        kwargs = transformers_model.format_inference_kwargs(
            OutlinesModelSettings(
                logit_bias={'20,21': 0.5, '22': 0.3, 'a': 0.2},  # type: ignore[reportArgumentType]
            )
        )
    assert 'logit_bias' not in kwargs

    # stop_strings from stop_sequences
    kwargs = transformers_model.format_inference_kwargs(OutlinesModelSettings(stop_sequences=['Paris', 'London']))
    assert kwargs['stop_strings'] == ['Paris', 'London']
    assert 'stop_sequences' not in kwargs

    # extra_body merging
    kwargs = transformers_model.format_inference_kwargs(
        OutlinesModelSettings(
            extra_body={'tokenizer': 'test_tokenizer'},
            max_tokens=100,
        )
    )
    assert kwargs['tokenizer'] == 'test_tokenizer'
    assert kwargs['max_tokens'] == 100
    assert 'extra_body' not in kwargs


@skip_if_llama_cpp_imports_unsuccessful
def test_model_settings_llamacpp(llamacpp_model: 'OutlinesModel') -> None:
    # unsupported arguments removed with warnings
    with pytest.warns(UserWarning, match='The llama_cpp model does not support'):
        kwargs = llamacpp_model.format_inference_kwargs(
            OutlinesModelSettings(
                timeout=1,
                parallel_tool_calls=True,
                extra_headers={'Authorization': 'Bearer 123'},
            )
        )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'extra_headers' not in kwargs

    # stop from stop_sequences
    kwargs = llamacpp_model.format_inference_kwargs(OutlinesModelSettings(stop_sequences=['Paris', 'London']))
    assert kwargs['stop'] == ['Paris', 'London']
    assert 'stop_sequences' not in kwargs
    assert 'stop' in kwargs

    # logit_bias with invalid keys
    with pytest.warns(UserWarning, match='The llama_cpp model expects the keys of the `logits_bias`'):
        kwargs = llamacpp_model.format_inference_kwargs(
            OutlinesModelSettings(
                logit_bias={'a': 0.5, 'b': 0.3}  # type: ignore[reportArgumentType]
            )
        )
    assert 'logit_bias' not in kwargs

    # logit_bias with valid integer keys
    kwargs = llamacpp_model.format_inference_kwargs(
        OutlinesModelSettings(
            logit_bias={20: 0.5, 30: 0.3}  # type: ignore[reportArgumentType]
        )
    )
    assert kwargs['logit_bias'] == {20: 0.5, 30: 0.3}


@skip_if_mlxlm_imports_unsuccessful
def test_model_settings_mlxlm(mlxlm_model: 'OutlinesModel') -> None:
    # all arguments are removed with warnings
    with pytest.warns(UserWarning, match='The mlxlm model does not support'):
        kwargs = mlxlm_model.format_inference_kwargs(
            OutlinesModelSettings(
                temperature=0.7,
                top_p=0.9,
                timeout=1,
                parallel_tool_calls=True,
                seed=123,
                presence_penalty=0.7,
                frequency_penalty=0.3,
                logit_bias={'20': 0.5},  # type: ignore[reportArgumentType]
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


@skip_if_sglang_imports_unsuccessful
def test_model_settings_sglang(sglang_model: 'OutlinesModel') -> None:
    # unsupported arguments removed with warnings
    with pytest.warns(UserWarning, match='The SgLang model does not support'):
        kwargs = sglang_model.format_inference_kwargs(
            OutlinesModelSettings(
                timeout=1,
                parallel_tool_calls=True,
                seed=123,
                logit_bias={'20': 10},
                extra_headers={'Authorization': 'Bearer 123'},
            )
        )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'seed' not in kwargs
    assert 'logit_bias' not in kwargs
    assert 'extra_headers' not in kwargs

    # stop from stop_sequences
    kwargs = sglang_model.format_inference_kwargs(OutlinesModelSettings(stop_sequences=['Paris', 'London']))
    assert kwargs['stop'] == ['Paris', 'London']
    assert 'stop_sequences' not in kwargs
    assert 'stop' in kwargs


@skip_if_vllm_imports_unsuccessful
def test_model_settings_vllm_offline(vllm_model_offline: 'OutlinesModel') -> None:
    # unsupported arguments removed with warnings
    with pytest.warns(UserWarning, match='The SgLang model does not support'):
        kwargs = vllm_model_offline.format_inference_kwargs(
            OutlinesModelSettings(
                timeout=1,
                parallel_tool_calls=True,
                extra_headers={'Authorization': 'Bearer 123'},
            )
        )
    assert 'timeout' not in kwargs
    assert 'parallel_tool_calls' not in kwargs
    assert 'extra_headers' not in kwargs

    # stop from stop_sequences
    kwargs = vllm_model_offline.format_inference_kwargs(OutlinesModelSettings(stop_sequences=['Paris', 'London']))
    assert kwargs['stop'] == ['Paris', 'London']
    assert 'stop_sequences' not in kwargs
    assert 'stop' in kwargs

    # special keys are preserved and others are in sampling params
    kwargs = vllm_model_offline.format_inference_kwargs(
        OutlinesModelSettings(  # type: ignore[reportCallIssue]
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
