from __future__ import annotations as _annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Iterable
from datetime import datetime
from types import ModuleType
from typing import Any

import httpx
import pytest
from devtools import debug
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_mock import MockerFixture

from pydantic_ai.messages import Message, ModelAnyResponse, ModelTextResponse
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from tests.conftest import ClientWithHandler


def find_filter_examples() -> Iterable[CodeExample]:
    for ex in find_examples('docs', 'pydantic_ai'):
        if ex.path.name != '_utils.py':
            yield ex


@pytest.mark.parametrize('example', find_filter_examples(), ids=str)
def test_docs_examples(
    example: CodeExample, eval_example: EvalExample, mocker: MockerFixture, client_with_handler: ClientWithHandler
):
    if example.path.name == '_utils.py':
        return
    # debug(example)
    mocker.patch('pydantic_ai.agent.models.infer_model', side_effect=mock_infer_model)
    mocker.patch('pydantic_ai._utils.datetime', MockedDatetime)

    mocker.patch('httpx.Client.get', side_effect=http_request)
    mocker.patch('httpx.Client.post', side_effect=http_request)
    mocker.patch('httpx.AsyncClient.get', side_effect=async_http_request)
    mocker.patch('httpx.AsyncClient.post', side_effect=async_http_request)

    ruff_ignore: list[str] = ['D']
    if str(example.path).endswith('docs/index.md'):
        ruff_ignore.append('F841')
    eval_example.set_config(ruff_ignore=ruff_ignore)

    call_name = 'main'
    if 'def test_application_code' in example.source:
        call_name = 'test_application_code'

    if eval_example.update_examples:
        eval_example.format(example)
        module_dict = eval_example.run_print_update(example, call=call_name)
    else:
        eval_example.lint(example)
        module_dict = eval_example.run_print_check(example, call=call_name)

    if example.path.name == 'dependencies.md' and 'title="joke_app.py"' in example.prefix:
        sys.modules['joke_app'] = module = ModuleType('joke_app')
        module.__dict__.update(module_dict)


def http_request(url: str, **kwargs: Any) -> httpx.Response:
    # sys.stdout.write(f'GET {args=} {kwargs=}\n')
    request = httpx.Request('GET', url, **kwargs)
    return httpx.Response(status_code=202, content='', request=request)


async def async_http_request(url: str, **kwargs: Any) -> httpx.Response:
    return http_request(url, **kwargs)


async def model_logic(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
    m = messages[-1]
    if m.role == 'user' and m.content == 'What is the weather like in West London and in Wiltshire?':
        return ModelTextResponse(content='The weather in West London is raining, while in Wiltshire it is sunny.')
    if m.role == 'user' and m.content == 'Tell me a joke.':
        return ModelTextResponse(content='Did you hear about the toothpaste scandal? They called it Colgate.')
    if m.role == 'user' and m.content == 'Explain?':
        return ModelTextResponse(content='This is an excellent joke invent by Samuel Colvin, it needs no explanation.')
    if m.role == 'user' and m.content == 'What is the capital of France?':
        return ModelTextResponse(content='Paris')
    else:
        sys.stdout.write(str(debug.format(messages, info)))
        raise RuntimeError(f'Unexpected message: {m}')


async def stream_model_logic(messages: list[Message], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    m = messages[-1]
    if m.role == 'user' and m.content == 'Tell me a joke.':
        *words, last_word = 'Did you hear about the toothpaste scandal? They called it Colgate.'.split(' ')
        for work in words:
            yield f'{work} '
            await asyncio.sleep(0.05)
        yield last_word
    else:
        sys.stdout.write(str(debug.format(messages, info)))
        raise RuntimeError(f'Unexpected message: {m}')


def mock_infer_model(_model: Model | KnownModelName) -> Model:
    return FunctionModel(model_logic, stream_function=stream_model_logic)


class MockedDatetime(datetime):
    @classmethod
    def now(cls, tz: Any = None) -> datetime:  # type: ignore
        return datetime(2032, 1, 2, 3, 4, 5, 6, tzinfo=tz)
