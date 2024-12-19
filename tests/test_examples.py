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

from pydantic_ai._utils import group_by_temporal
from pydantic_ai.messages import (
    ArgsDict,
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

from .conftest import ClientWithHandler, TestEnv

try:
    from pydantic_ai.models.vertexai import VertexAIModel
except ImportError:
    VertexAIModel = None


try:
    import logfire
except ImportError:
    logfire = None


pytestmark = pytest.mark.skipif(VertexAIModel is None or logfire is None, reason='google-auth or logfire not installed')


def find_filter_examples() -> Iterable[CodeExample]:
    for ex in find_examples('docs', 'pydantic_ai_slim'):
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
    set_event_loop: None,
):
    mocker.patch('pydantic_ai.agent.models.infer_model', side_effect=mock_infer_model)
    mocker.patch('pydantic_ai._utils.group_by_temporal', side_effect=mock_group_by_temporal)
    mocker.patch('pydantic_ai.models.vertexai._creds_from_file', return_value=MockCredentials())

    mocker.patch('httpx.Client.get', side_effect=http_request)
    mocker.patch('httpx.Client.post', side_effect=http_request)
    mocker.patch('httpx.AsyncClient.get', side_effect=async_http_request)
    mocker.patch('httpx.AsyncClient.post', side_effect=async_http_request)
    mocker.patch('random.randint', return_value=4)

    env.set('OPENAI_API_KEY', 'testing')
    env.set('GEMINI_API_KEY', 'testing')
    env.set('GROQ_API_KEY', 'testing')

    sys.path.append('tests/example_modules')

    prefix_settings = example.prefix_settings()
    opt_title = prefix_settings.get('title')
    opt_test = prefix_settings.get('test', '')
    opt_lint = prefix_settings.get('lint', '')
    cwd = Path.cwd()

    if opt_test.startswith('skip') and opt_lint.startswith('skip'):
        pytest.skip('both running code and lint skipped')

    if opt_title == 'sql_app_evals.py':
        os.chdir(tmp_path)
        examples = [{'request': f'sql prompt {i}', 'sql': f'SELECT {i}'} for i in range(15)]
        with (tmp_path / 'examples.json').open('w') as f:
            json.dump(examples, f)

    ruff_ignore: list[str] = ['D']
    # `from bank_database import DatabaseConn` wrongly sorted in imports
    # waiting for https://github.com/pydantic/pytest-examples/issues/43
    # and https://github.com/pydantic/pytest-examples/issues/46
    if opt_lint == 'not-imports' or 'import DatabaseConn' in example.source:
        ruff_ignore.append('I001')

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
        pytest.skip(opt_test[4:].lstrip(' -') or 'running code skipped')
    else:
        if eval_example.update_examples:
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
    return re.sub(r'datetime.date\(', 'date(', s)


def http_request(url: str, **kwargs: Any) -> httpx.Response:
    # sys.stdout.write(f'GET {args=} {kwargs=}\n')
    request = httpx.Request('GET', url, **kwargs)
    return httpx.Response(status_code=202, content='', request=request)


async def async_http_request(url: str, **kwargs: Any) -> httpx.Response:
    return http_request(url, **kwargs)


text_responses: dict[str, str | ToolCallPart] = {
    'What is the weather like in West London and in Wiltshire?': (
        'The weather in West London is raining, while in Wiltshire it is sunny.'
    ),
    'Tell me a joke.': 'Did you hear about the toothpaste scandal? They called it Colgate.',
    'Explain?': 'This is an excellent joke invent by Samuel Colvin, it needs no explanation.',
    'What is the capital of France?': 'Paris',
    'What is the capital of Italy?': 'Rome',
    'What is the capital of the UK?': 'London',
    'Who was Albert Einstein?': 'Albert Einstein was a German-born theoretical physicist.',
    'What was his most famous equation?': "Albert Einstein's most famous equation is (E = mc^2).",
    'What is the date?': 'Hello Frank, the date today is 2032-01-02.',
    'Put my money on square eighteen': ToolCallPart(tool_name='roulette_wheel', args=ArgsDict({'square': 18})),
    'I bet five is the winner': ToolCallPart(tool_name='roulette_wheel', args=ArgsDict({'square': 5})),
    'My guess is 4': ToolCallPart(tool_name='roll_die', args=ArgsDict({})),
    'Send a message to John Doe asking for coffee next week': ToolCallPart(
        tool_name='get_user_by_name', args=ArgsDict({'name': 'John'})
    ),
    'Please get me the volume of a box with size 6.': ToolCallPart(tool_name='calc_volume', args=ArgsDict({'size': 6})),
    'Where does "hello world" come from?': (
        'The first known use of "hello, world" was in a 1974 textbook about the C programming language.'
    ),
    'What is my balance?': ToolCallPart(tool_name='customer_balance', args=ArgsDict({'include_pending': True})),
    'I just lost my card!': ToolCallPart(
        tool_name='final_result',
        args=ArgsDict(
            {
                'support_advice': (
                    "I'm sorry to hear that, John. "
                    'We are temporarily blocking your card to prevent unauthorized transactions.'
                ),
                'block_card': True,
                'risk': 8,
            }
        ),
    ),
    'Where were the olympics held in 2012?': ToolCallPart(
        tool_name='final_result',
        args=ArgsDict({'city': 'London', 'country': 'United Kingdom'}),
    ),
    'The box is 10x20x30': 'Please provide the units for the dimensions (e.g., cm, in, m).',
    'The box is 10x20x30 cm': ToolCallPart(
        tool_name='final_result',
        args=ArgsDict({'width': 10, 'height': 20, 'depth': 30, 'units': 'cm'}),
    ),
    'red square, blue circle, green triangle': ToolCallPart(
        tool_name='final_result_list',
        args=ArgsDict({'response': ['red', 'blue', 'green']}),
    ),
    'square size 10, circle size 20, triangle size 30': ToolCallPart(
        tool_name='final_result_list_2',
        args=ArgsDict({'response': [10, 20, 30]}),
    ),
    'get me uses who were last active yesterday.': ToolCallPart(
        tool_name='final_result_Success',
        args=ArgsDict({'sql_query': 'SELECT * FROM users WHERE last_active::date = today() - interval 1 day'}),
    ),
    'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.': ToolCallPart(
        tool_name='final_result',
        args=ArgsDict(
            {
                'name': 'Ben',
                'dob': '1990-01-28',
                'bio': 'Likes the chain the dog and the pyramid',
            }
        ),
    ),
    'What is the capital of Italy? Answer with just the city.': 'Rome',
    'What is the capital of Italy? Answer with a paragraph.': (
        'The capital of Italy is Rome (Roma, in Italian), which has been a cultural and political center for centuries.'
        'Rome is known for its rich history, stunning architecture, and delicious cuisine.'
    ),
    'Begin infinite retry loop!': ToolCallPart(tool_name='infinite_retry_tool', args=ArgsDict({})),
}


async def model_logic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
    m = messages[-1].parts[-1]
    if isinstance(m, UserPromptPart):
        if response := text_responses.get(m.content):
            if isinstance(response, str):
                return ModelResponse.from_text(content=response)
            else:
                return ModelResponse(parts=[response])

        if re.fullmatch(r'sql prompt \d+', m.content):
            return ModelResponse.from_text(content='SELECT 1')

    elif isinstance(m, ToolReturnPart) and m.tool_name == 'roulette_wheel':
        win = m.content == 'winner'
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=ArgsDict({'response': win}))])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'roll_die':
        return ModelResponse(parts=[ToolCallPart(tool_name='get_player_name', args=ArgsDict({}))])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_player_name':
        return ModelResponse.from_text(content="Congratulations Anne, you guessed correctly! You're a winner!")
    if (
        isinstance(m, RetryPromptPart)
        and isinstance(m.content, str)
        and m.content.startswith("No user found with name 'Joh")
    ):
        return ModelResponse(parts=[ToolCallPart(tool_name='get_user_by_name', args=ArgsDict({'name': 'John Doe'}))])
    elif isinstance(m, RetryPromptPart) and m.tool_name == 'infinite_retry_tool':
        return ModelResponse(parts=[ToolCallPart(tool_name='infinite_retry_tool', args=ArgsDict({}))])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'get_user_by_name':
        args = {
            'message': 'Hello John, would you be free for coffee sometime next week? Let me know what works for you!',
            'user_id': 123,
        }
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=ArgsDict(args))])
    elif isinstance(m, RetryPromptPart) and m.tool_name == 'calc_volume':
        return ModelResponse(parts=[ToolCallPart(tool_name='calc_volume', args=ArgsDict({'size': 6}))])
    elif isinstance(m, ToolReturnPart) and m.tool_name == 'customer_balance':
        args = {
            'support_advice': 'Hello John, your current account balance, including pending transactions, is $123.45.',
            'block_card': False,
            'risk': 1,
        }
        return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args=ArgsDict(args))])
    else:
        sys.stdout.write(str(debug.format(messages, info)))
        raise RuntimeError(f'Unexpected message: {m}')


async def stream_model_logic(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:  # pragma: no cover
    m = messages[-1].parts[-1]
    if isinstance(m, UserPromptPart):
        if response := text_responses.get(m.content):
            if isinstance(response, str):
                words = response.split(' ')
                chunk: list[str] = []
                for work in words:
                    chunk.append(work)
                    if len(chunk) == 3:
                        yield ' '.join(chunk) + ' '
                        chunk.clear()
                if chunk:
                    yield ' '.join(chunk)
                return
            else:
                json_text = response.args_as_json_str()

                yield {1: DeltaToolCall(name=response.tool_name)}
                for chunk_index in range(0, len(json_text), 15):
                    text_chunk = json_text[chunk_index : chunk_index + 15]
                    yield {1: DeltaToolCall(json_args=text_chunk)}
                return

    sys.stdout.write(str(debug.format(messages, info)))
    raise RuntimeError(f'Unexpected message: {m}')


def mock_infer_model(model: Model | KnownModelName) -> Model:
    if isinstance(model, (FunctionModel, TestModel)):
        return model
    elif model == 'test':
        return TestModel()
    else:
        return FunctionModel(model_logic, stream_function=stream_model_logic)


def mock_group_by_temporal(aiter: Any, soft_max_interval: float | None) -> Any:
    """Mock group_by_temporal to avoid debouncing, since the iterators above have no delay."""
    return group_by_temporal(aiter, None)


@dataclass
class MockCredentials:
    project_id = 'foobar'
