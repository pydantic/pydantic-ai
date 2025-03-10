from __future__ import annotations as _annotations

import json
import os
import re
import sys
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from devtools import debug
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_mock import MockerFixture

from pydantic_ai import ModelHTTPError
from pydantic_ai._utils import group_by_temporal
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from .conftest import ClientWithHandler, TestEnv

try:
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider
except ImportError:
    GoogleVertexProvider = None


try:
    import logfire
except ImportError:
    logfire = None


pytestmark = pytest.mark.skipif(
    GoogleVertexProvider is None or logfire is None, reason='google-auth or logfire not installed'
)


def find_filter_examples() -> Iterable[CodeExample]:
    for ex in find_examples('docs', 'pydantic_ai_slim', 'pydantic_graph'):
        if ex.path.name != '_utils.py':
            yield ex


@pytest.mark.parametrize('example', find_filter_examples(), ids=str)
def test_docs_examples(
    example: CodeExample,
    eval_example: EvalExample,
    mocker: MockerFixture,
    client_with_handler: ClientWithHandler,
    env: TestEnv,
    tmp_path: Path,
):
    mocker.patch('pydantic_ai.agent.models.infer_model', side_effect=mock_infer_model)
    mocker.patch('pydantic_ai._utils.group_by_temporal', side_effect=mock_group_by_temporal)
    mocker.patch('pydantic_ai.models.vertexai._creds_from_file', return_value=MockCredentials())

    mocker.patch('httpx.Client.get', side_effect=http_request)
    mocker.patch('httpx.Client.post', side_effect=http_request)
    mocker.patch('httpx.AsyncClient.get', side_effect=async_http_request)
    mocker.patch('httpx.AsyncClient.post', side_effect=async_http_request)
    mocker.patch('random.randint', return_value=4)
    mocker.patch('rich.prompt.Prompt.ask', side_effect=rich_prompt_ask)

    env.set('OPENAI_API_KEY', 'testing')
    env.set('GEMINI_API_KEY', 'testing')
    env.set('GROQ_API_KEY', 'testing')
    env.set('CO_API_KEY', 'testing')

    sys.path.append('tests/example_modules')

    prefix_settings = example.prefix_settings()
    opt_title = prefix_settings.get('title')
    opt_test = prefix_settings.get('test', '')
    opt_lint = prefix_settings.get('lint', '')
    noqa = prefix_settings.get('noqa', '')
    python_version = prefix_settings.get('py', None)

    if python_version:
        python_version_info = tuple(int(v) for v in python_version.split('.'))
        if sys.version_info < python_version_info:
            pytest.skip(f'Python version {python_version} required')

    cwd = Path.cwd()

    if opt_test.startswith('skip') and opt_lint.startswith('skip'):
        pytest.skip('both running code and lint skipped')

    if opt_title == 'sql_app_evals.py':
        os.chdir(tmp_path)
        examples = [{'request': f'sql prompt {i}', 'sql': f'SELECT {i}'} for i in range(15)]
        with (tmp_path / 'examples.json').open('w') as f:
            json.dump(examples, f)

    ruff_ignore: list[str] = ['D', 'Q001']
    # `from bank_database import DatabaseConn` wrongly sorted in imports
    # waiting for https://github.com/pydantic/pytest-examples/issues/43
    # and https://github.com/pydantic/pytest-examples/issues/46
    if 'import DatabaseConn' in example.source:
        ruff_ignore.append('I001')

    if noqa:
        ruff_ignore.extend(noqa.upper().split())

    line_length = int(prefix_settings.get('line_length', '88'))

    eval_example.set_config(ruff_ignore=ruff_ignore, target_version='py39', line_length=line_length)
    eval_example.print_callback = print_callback

    call_name = prefix_settings.get('call_name', 'main')

    if not opt_lint.startswith('skip'):
        if eval_example.update_examples:  # pragma: no cover
            eval_example.format(example)
        else:
            eval_example.lint(example)

    if opt_test.startswith('skip'):
        print(opt_test[4:].lstrip(' -') or 'running code skipped')
    else:
        if eval_example.update_examples:  # pragma: no cover
            module_dict = eval_example.run_print_update(example, call=call_name)
        else:
            module_dict = eval_example.run_print_check(example, call=call_name)

        os.chdir(cwd)
        if title := opt_title:
            if title.endswith('.py'):
                module_name = title[:-3]
                sys.modules[module_name] = module = ModuleType(module_name)
                module.__dict__.update(module_dict)


def print_callback(s: str) -> str:
    s = re.sub(r'datetime\.datetime\(.+?\)', 'datetime.datetime(...)', s, flags=re.DOTALL)
    s = re.sub(r'\d\.\d{4,}e-0\d', '0.0...', s)
    return re.sub(r'datetime.date\(', 'date(', s)


def http_request(url: str, **kwargs: Any) -> httpx.Response:
    # sys.stdout.write(f'GET {args=} {kwargs=}\n')
    request = httpx.Request('GET', url, **kwargs)
    return httpx.Response(status_code=202, content='', request=request)


async def async_http_request(url: str, **kwargs: Any) -> httpx.Response:
    return http_request(url, **kwargs)


def rich_prompt_ask(prompt: str, *_args: Any, **_kwargs: Any) -> str:
    if prompt == 'Where would you like to fly from and to?':
        return 'SFO to ANC'
    elif prompt == 'What seat would you like?':
        return 'window seat with leg room'
    if prompt == 'Insert coins':
        return '1'
    elif prompt == 'Select product':
        return 'crisps'
    elif prompt == 'What is the capital of France?':
        return 'Vichy'
    elif prompt == 'what is 1 + 1?':
        return '2'
    else:  # pragma: no cover
        raise ValueError(f'Unexpected prompt: {prompt}')


text_responses: dict[str, str | ToolCallPart] = {
    'What is the weather like in West London and in Wiltshire?': (
        'The weather in West London is raining, while in Wiltshire it is sunny.'
    ),
    'What will the weather be like in Paris on Tuesday?': ToolCallPart(
        tool_name='weather_forecast', args={'location': 'Paris', 'forecast_date': '2030-01-01'}, tool_call_id='0001'
    ),
    'Tell me a joke.': 'Did you hear about the toothpaste scandal? They called it Colgate.',
    'Tell me a different joke.': 'No.',
    'Explain?': 'This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
    'What is the capital of France?': 'Paris',
    'What is the capital of Italy?': 'Rome',
    'What is the capital of the UK?': 'London',
    'Who was Albert Einstein?': 'Albert Einstein was a German-born theoretical physicist.',
    'What was his most famous equation?': "Albert Einstein's most famous equation is (E = mc^2).",
    'What is the date?': 'Hello Frank, the date today is 2032-01-02.',
    'Put my money on square eighteen': ToolCallPart(tool_name='roulette_wheel', args={'square': 18}),
    'I bet five is the winner': ToolCallPart(tool_name='roulette_wheel', args={'square': 5}),
    'My guess is 4': ToolCallPart(tool_name='roll_die', args={}),
    'Send a message to John Doe asking for coffee next week': ToolCallPart(
        tool_name='get_user_by_name', args={'name': 'John'}
    ),
    'Please get me the volume of a box with size 6.': ToolCallPart(tool_name='calc_volume', args={'size': 6}),
    'Where does "hello world" come from?': (
        'The first known use of "hello, world" was in a 1974 textbook about the C programming language.'
    ),
    'What is my balance?': ToolCallPart(tool_name='customer_balance', args={'include_pending': True}),
    'I just lost my card!': ToolCallPart(
        tool_name='final_result',
        args={
            'support_advice': (
                "I'm sorry to hear that, John. "
                'We are temporarily blocking your card to prevent unauthorized transactions.'
            ),
            'block_card': True,
            'risk': 8,
        },
    ),
    'Where were the olympics held in 2012?': ToolCallPart(
        tool_name='final_result',
        args={'city': 'London', 'country': 'United Kingdom'},
    ),
    'The box is 10x20x30': 'Please provide the units for the dimensions (e.g., cm, in, m).',
    'The box is 10x20x30 cm': ToolCallPart(
        tool_name='final_result',
        args={'width': 10, 'height': 20, 'depth': 30, 'units': 'cm'},
    ),
    'red square, blue circle, green triangle': ToolCallPart(
        tool_name='final_result_list',
        args={'response': ['red', 'blue', 'green']},
    ),
    'square size 10, circle size 20, triangle size 30': ToolCallPart(
        tool_name='final_result_list_2',
        args={'response': [10, 20, 30]},
    ),
    'get me users who were last active yesterday.': ToolCallPart(
        tool_name='final_result_Success',
        args={'sql_query': 'SELECT * FROM users WHERE last_active::date = today() - interval 1 day'},
    ),
    'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.': ToolCallPart(
        tool_name='final_result',
        args={
            'name': 'Ben',
            'dob': '1990-01-28',
            'bio': 'Likes the chain the dog and the pyramid',
        },
    ),
    'What is the capital of Italy? Answer with just the city.': 'Rome',
    'What is the capital of Italy? Answer with a paragraph.': (
        'The capital of Italy is Rome (Roma, in Italian), which has been a cultural and political center for centuries.'
        'Rome is known for its rich history, stunning architecture, and delicious cuisine.'
    ),
    'Begin infinite retry loop!': ToolCallPart(tool_name='infinite_retry_tool', args={}),
    'Please generate 5 jokes.': ToolCallPart(
        tool_name='final_result',
        args={'response': []},
    ),
    'SFO to ANC': ToolCallPart(
        tool_name='flight_search',
        args={'origin': 'SFO', 'destination': 'ANC'},
    ),
    'window seat with leg room': ToolCallPart(
        tool_name='final_result_SeatPreference',
        args={'row': 1, 'seat': 'A'},
    ),
    'Ask a simple question with a single correct answer.': 'What is the capital of France?',
    '<examples>\n  <question>What is the capital of France?</question>\n  <answer>Vichy</answer>\n</examples>': ToolCallPart(
        tool_name='final_result',
        args={'correct': False, 'comment': 'Vichy is no longer the capital of France.'},
    ),
    '<examples>\n  <question>what is 1 + 1?</question>\n  <answer>2</answer>\n</examples>': ToolCallPart(
        tool_name='final_result',
        args={'correct': True, 'comment': 'Well done, 1 + 1 = 2'},
    ),
}

tool_responses: dict[tuple[str, str], str] = {
    (
        'weather_forecast',
        'The forecast in Paris on 2030-01-01 is 24°C and sunny.',
    ): 'It will be warm and sunny in Paris on Tuesday.',
}


async def model_logic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover  # noqa: C901
    m = messages[-1].parts[-1]
    if isinstance(m, UserPromptPart):
        assert isinstance(m.content, str)
        if m.content == 'Tell me a joke.' and any(t.name == 'joke_factory' for t in info.function_tools):
            return ModelResponse(parts=[ToolCallPart(tool_name='joke_factory', args={'count': 5})])
        elif m.content == 'Please generate 5 jokes.' and any(t.name == 'get_jokes' for t in info.function_tools):
            return ModelResponse(parts=[ToolCallPart(tool_name='get_jokes', args={'count': 5})])
        elif re.fullmatch(r'sql prompt \d+', m.content):
            return ModelResponse(parts=[TextPart('SELECT 1')])
        elif m.content.startswith('Write a welcome email for the user:'):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={
                            'subject': 'Welcome to our tech blog!',
                            'body': 'Hello John, Welcome to our tech blog! ...',
                        },
                    )
                ]
            )
        elif m.content.startswith('Write a list of 5 very rude things that I might say'):
            raise UnexpectedModelBehavior('Safety settings triggered', body='<safety settings details>')
        elif m.content.startswith('<examples>\n  <user>'):
            return ModelResponse(parts=[ToolCallPart(tool_name='final_result_EmailOk', args={})])
        elif m.content == 'Ask a simple question with a single correct answer.' and len(messages) > 2:
            return ModelResponse(parts=[TextPart('what is 1 + 1?')])
        elif response := text_responses.get(m.content):
            if isinstance(response, str):
                return ModelResponse(parts=[TextPart(response)])
            else:
                return ModelResponse(parts=[response])

    elif isinstance(m, ToolReturnPart) and m.tool_name == 'roulette_wheel':
        win = m.content == 'winner'
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'response': win})])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'roll_die':
        return ModelResponse(parts=[ToolCallPart(tool_name='get_player_name', args={})])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_player_name':
        return ModelResponse(parts=[TextPart("Congratulations Anne, you guessed correctly! You're a winner!")])
    if (
        isinstance(m, RetryPromptPart)
        and isinstance(m.content, str)
        and m.content.startswith("No user found with name 'Joh")
    ):
        return ModelResponse(parts=[ToolCallPart(tool_name='get_user_by_name', args={'name': 'John Doe'})])
    elif isinstance(m, RetryPromptPart) and m.tool_name == 'infinite_retry_tool':
        return ModelResponse(parts=[ToolCallPart(tool_name='infinite_retry_tool', args={})])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_user_by_name':
        args: dict[str, Any] = {
            'message': 'Hello John, would you be free for coffee sometime next week? Let me know what works for you!',
            'user_id': 123,
        }
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=args)])
    elif isinstance(m, RetryPromptPart) and m.tool_name == 'calc_volume':
        return ModelResponse(parts=[ToolCallPart(tool_name='calc_volume', args={'size': 6})])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'customer_balance':
        args = {
            'support_advice': 'Hello John, your current account balance, including pending transactions, is $123.45.',
            'block_card': False,
            'risk': 1,
        }
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=args)])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'joke_factory':
        return ModelResponse(parts=[TextPart('Did you hear about the toothpaste scandal? They called it Colgate.')])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_jokes':
        args = {'response': []}
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=args)])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'flight_search':
        args = {'flight_number': m.content.flight_number}  # type: ignore
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result_FlightDetails', args=args)])
    else:
        sys.stdout.write(str(debug.format(messages, info)))
        raise RuntimeError(f'Unexpected message: {m}')


async def stream_model_logic(  # noqa C901
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:  # pragma: no cover
    async def stream_text_response(r: str) -> AsyncIterator[str]:
        if isinstance(r, str):
            words = r.split(' ')
            chunk: list[str] = []
            for word in words:
                chunk.append(word)
                if len(chunk) == 3:
                    yield ' '.join(chunk) + ' '
                    chunk.clear()
            if chunk:
                yield ' '.join(chunk)

    async def stream_tool_call_response(r: ToolCallPart) -> AsyncIterator[DeltaToolCalls]:
        json_text = r.args_as_json_str()

        yield {1: DeltaToolCall(name=r.tool_name, tool_call_id=r.tool_call_id)}
        for chunk_index in range(0, len(json_text), 15):
            text_chunk = json_text[chunk_index : chunk_index + 15]
            yield {1: DeltaToolCall(json_args=text_chunk)}

    async def stream_part_response(r: str | ToolCallPart) -> AsyncIterator[str | DeltaToolCalls]:
        if isinstance(r, str):
            async for chunk in stream_text_response(r):
                yield chunk
        else:
            async for chunk in stream_tool_call_response(r):
                yield chunk

    last_part = messages[-1].parts[-1]
    if isinstance(last_part, UserPromptPart):
        assert isinstance(last_part.content, str)
        if response := text_responses.get(last_part.content):
            async for chunk in stream_part_response(response):
                yield chunk
            return
    elif isinstance(last_part, ToolReturnPart):
        assert isinstance(last_part.content, str)
        if response := tool_responses.get((last_part.tool_name, last_part.content)):
            async for chunk in stream_part_response(response):
                yield chunk
            return

    sys.stdout.write(str(debug.format(messages, info)))
    raise RuntimeError(f'Unexpected message: {last_part}')


def mock_infer_model(model: Model | KnownModelName) -> Model:
    if model == 'test':
        return TestModel()

    if isinstance(model, str):
        # Use the non-mocked model inference to ensure we get the same model name the user would
        model = infer_model(model)

    if isinstance(model, FallbackModel):
        # When a fallback model is encountered, replace any OpenAIModel with a model that will raise a ModelHTTPError.
        # Otherwise, do the usual inference.
        def raise_http_error(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
            raise ModelHTTPError(401, 'Invalid API Key')

        mock_fallback_models: list[Model] = []
        for m in model.models:
            try:
                from pydantic_ai.models.openai import OpenAIModel
            except ImportError:
                OpenAIModel = type(None)

            if isinstance(m, OpenAIModel):
                # Raise an HTTP error for OpenAIModel
                mock_fallback_models.append(FunctionModel(raise_http_error, model_name=m.model_name))
            else:
                mock_fallback_models.append(mock_infer_model(m))
        return FallbackModel(*mock_fallback_models)
    if isinstance(model, (FunctionModel, TestModel)):
        return model
    else:
        model_name = model if isinstance(model, str) else model.model_name
        return FunctionModel(model_logic, stream_function=stream_model_logic, model_name=model_name)


def mock_group_by_temporal(aiter: Any, soft_max_interval: float | None) -> Any:
    """Mock group_by_temporal to avoid debouncing, since the iterators above have no delay."""
    return group_by_temporal(aiter, None)


@dataclass
class MockCredentials:
    project_id = 'foobar'
