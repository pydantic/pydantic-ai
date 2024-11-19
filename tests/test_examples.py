from __future__ import annotations as _annotations

import re
import sys
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import httpx
import pydantic_core
import pytest
from devtools import debug
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_mock import MockerFixture

from pydantic_ai._utils import group_by_temporal
from pydantic_ai.messages import (
    ArgsObject,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    ToolCall,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from tests.conftest import ClientWithHandler


@pytest.fixture(scope='module', autouse=True)
def register_fake_db():
    class FakeTable:
        def get(self, name: str) -> int | None:
            if name == 'John Doe':
                return 123

    @dataclass
    class DatabaseConn:
        users: FakeTable = field(default_factory=FakeTable)

        async def execute(self, query: str) -> None:
            pass

    class QueryError(RuntimeError):
        pass

    module_name = 'fake_database'
    sys.modules[module_name] = module = ModuleType(module_name)
    module.__dict__.update({'DatabaseConn': DatabaseConn, 'QueryError': QueryError})

    yield

    sys.modules.pop(module_name)


@pytest.fixture(scope='module', autouse=True)
def register_bank_db():
    class DatabaseConn:
        @classmethod
        async def customer_name(cls, *, id: int) -> str | None:
            if id == 123:
                return 'John'

        @classmethod
        async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
            if id == 123:
                return 123.45
            else:
                raise ValueError('Customer not found')

    module_name = 'bank_database'
    sys.modules[module_name] = module = ModuleType(module_name)
    module.__dict__.update({'DatabaseConn': DatabaseConn})

    yield

    sys.modules.pop(module_name)


def find_filter_examples() -> Iterable[CodeExample]:
    for ex in find_examples('docs', 'pydantic_ai'):
        if ex.path.name != '_utils.py':
            yield ex


@pytest.mark.parametrize('example', find_filter_examples(), ids=str)
def test_docs_examples(
    example: CodeExample, eval_example: EvalExample, mocker: MockerFixture, client_with_handler: ClientWithHandler
):
    # debug(example)
    mocker.patch('pydantic_ai.agent.models.infer_model', side_effect=mock_infer_model)
    mocker.patch('pydantic_ai._utils.group_by_temporal', side_effect=mock_group_by_temporal)

    mocker.patch('httpx.Client.get', side_effect=http_request)
    mocker.patch('httpx.Client.post', side_effect=http_request)
    mocker.patch('httpx.AsyncClient.get', side_effect=async_http_request)
    mocker.patch('httpx.AsyncClient.post', side_effect=async_http_request)
    mocker.patch('random.randint', return_value=4)

    prefix_settings = example.prefix_settings()

    ruff_ignore: list[str] = ['D']
    # `from bank_database import DatabaseConn` wrongly sorted in imports
    # waiting for https://github.com/pydantic/pytest-examples/issues/43
    if 'from bank_database import DatabaseConn' in example.source:
        ruff_ignore.append('I001')

    line_length = 88
    if prefix_settings.get('title') in ('streamed_hello_world.py', 'streamed_user_profile.py'):
        line_length = 120

    eval_example.set_config(ruff_ignore=ruff_ignore, target_version='py39', line_length=line_length)

    eval_example.print_callback = print_callback

    call_name = 'main'
    if 'def test_application_code' in example.source:
        call_name = 'test_application_code'

    if eval_example.update_examples:  # pragma: no cover
        eval_example.format(example)
        module_dict = eval_example.run_print_update(example, call=call_name)
    else:
        eval_example.lint(example)
        module_dict = eval_example.run_print_check(example, call=call_name)

    if title := prefix_settings.get('title'):
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


text_responses: dict[str, str | ToolCall] = {
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
    'Put my money on square eighteen': ToolCall(tool_name='roulette_wheel', args=ArgsObject({'square': 18})),
    'I bet five is the winner': ToolCall(tool_name='roulette_wheel', args=ArgsObject({'square': 5})),
    'My guess is 4': ToolCall(tool_name='roll_die', args=ArgsObject({})),
    'Send a message to John Doe asking for coffee next week': ToolCall(
        tool_name='get_user_by_name', args=ArgsObject({'name': 'John'})
    ),
    'Please get me the volume of a box with size 6.': ToolCall(tool_name='calc_volume', args=ArgsObject({'size': 6})),
    'Where does "hello world" come from?': (
        'The first known use of "hello, world" was in a 1974 textbook about the C programming language.'
    ),
    'What is my balance?': ToolCall(tool_name='customer_balance', args=ArgsObject({'include_pending': True})),
    'I just lost my card!': ToolCall(
        tool_name='final_result',
        args=ArgsObject(
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
    'Where the olympics held in 2012?': ToolCall(
        tool_name='final_result',
        args=ArgsObject({'city': 'London', 'country': 'United Kingdom'}),
    ),
    'The box is 10x20x30': 'Please provide the units for the dimensions (e.g., cm, in, m).',
    'The box is 10x20x30 cm': ToolCall(
        tool_name='final_result',
        args=ArgsObject({'width': 10, 'height': 20, 'depth': 30, 'units': 'cm'}),
    ),
    'red square, blue circle, green triangle': ToolCall(
        tool_name='final_result_list',
        args=ArgsObject({'response': ['red', 'blue', 'green']}),
    ),
    'square size 10, circle size 20, triangle size 30': ToolCall(
        tool_name='final_result_list_2',
        args=ArgsObject({'response': [10, 20, 30]}),
    ),
    'get me uses who were last active yesterday.': ToolCall(
        tool_name='final_result_Success',
        args=ArgsObject({'sql_query': 'SELECT * FROM users WHERE last_active::date = today() - interval 1 day'}),
    ),
    'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.': ToolCall(
        tool_name='final_result',
        args=ArgsObject(
            {
                'name': 'Ben',
                'dob': '1990-01-28',
                'bio': 'Likes the chain the dog and the pyramid',
            }
        ),
    ),
}


async def model_logic(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:  # pragma: no cover
    m = messages[-1]
    if m.role == 'user':
        if response := text_responses.get(m.content):
            if isinstance(response, str):
                return ModelTextResponse(content=response)
            else:
                return ModelStructuredResponse(calls=[response])

    elif m.role == 'tool-return' and m.tool_name == 'roulette_wheel':
        win = m.content == 'winner'
        return ModelStructuredResponse(calls=[ToolCall(tool_name='final_result', args=ArgsObject({'response': win}))])
    elif m.role == 'tool-return' and m.tool_name == 'roll_die':
        return ModelStructuredResponse(calls=[ToolCall(tool_name='get_player_name', args=ArgsObject({}))])
    elif m.role == 'tool-return' and m.tool_name == 'get_player_name':
        return ModelTextResponse(content="Congratulations Adam, you guessed correctly! You're a winner!")
    if m.role == 'retry-prompt' and isinstance(m.content, str) and m.content.startswith("No user found with name 'Joh"):
        return ModelStructuredResponse(
            calls=[ToolCall(tool_name='get_user_by_name', args=ArgsObject({'name': 'John Doe'}))]
        )
    elif m.role == 'tool-return' and m.tool_name == 'get_user_by_name':
        args = {
            'message': 'Hello John, would you be free for coffee sometime next week? Let me know what works for you!',
            'user_id': 123,
        }
        return ModelStructuredResponse(calls=[ToolCall(tool_name='final_result', args=ArgsObject(args))])
    elif m.role == 'retry-prompt' and m.tool_name == 'calc_volume':
        return ModelStructuredResponse(calls=[ToolCall(tool_name='calc_volume', args=ArgsObject({'size': 6}))])
    elif m.role == 'tool-return' and m.tool_name == 'customer_balance':
        args = {
            'support_advice': 'Hello John, your current account balance, including pending transactions, is $123.45.',
            'block_card': False,
            'risk': 1,
        }
        return ModelStructuredResponse(calls=[ToolCall(tool_name='final_result', args=ArgsObject(args))])
    else:
        sys.stdout.write(str(debug.format(messages, info)))
        raise RuntimeError(f'Unexpected message: {m}')


async def stream_model_logic(
    messages: list[Message], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:  # pragma: no cover
    m = messages[-1]
    if m.role == 'user':
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
                if isinstance(response.args, ArgsObject):
                    json_text = pydantic_core.to_json(response.args.args_object).decode()
                else:
                    json_text = response.args.args_json

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
