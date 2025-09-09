import json

import pytest
from pydantic import BaseModel

from pydantic_ai._output import ObjectOutputProcessor, OutputToolset
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ToolRetryError
from pydantic_ai.usage import RunUsage


class Name(BaseModel):
    name: str


def _run_ctx() -> RunContext[None]:
    # Minimal run context; model is unused in this code path
    return RunContext(deps=None, model=object(), usage=RunUsage())  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_output_processor_accepts_response_json_string_list_of_models():
    processor = ObjectOutputProcessor(list[Name])
    rc = _run_ctx()

    payload = {'response': json.dumps([{'name': 'John'}, {'name': 'Jane'}])}
    result = await processor.process(json.dumps(payload), rc)

    assert [n.name for n in result] == ['John', 'Jane']


@pytest.mark.anyio
async def test_output_processor_accepts_response_native_list_of_models():
    processor = ObjectOutputProcessor(list[Name])
    rc = _run_ctx()

    payload = {'response': [{'name': 'Alice'}, {'name': 'Bob'}]}
    result = await processor.process(json.dumps(payload), rc)

    assert [n.name for n in result] == ['Alice', 'Bob']


@pytest.mark.anyio
async def test_output_processor_accepts_python_dict_with_response_json_string():
    processor = ObjectOutputProcessor(list[Name])
    rc = _run_ctx()

    # Top-level args already parsed to dict, with inner response as JSON string
    args_dict = {'response': json.dumps([{'name': 'Ann'}, {'name': 'Ben'}])}
    result = await processor.process(args_dict, rc)

    assert [n.name for n in result] == ['Ann', 'Ben']


@pytest.mark.anyio
async def test_output_processor_invalid_response_json_string_raises():
    processor = ObjectOutputProcessor(list[Name])
    rc = _run_ctx()

    payload = {'response': 'not-json'}
    with pytest.raises(ToolRetryError):
        await processor.process(json.dumps(payload), rc)


@pytest.mark.anyio
async def test_output_toolset_validator_accepts_response_json_string_and_call():
    toolset = OutputToolset.build([list[Name]])
    assert toolset is not None
    tools = await toolset.get_tools(_run_ctx())
    assert 'final_result' in tools
    tool = tools['final_result']

    # Validator accepts JSON string for nested `response`
    args_dict = tool.args_validator.validate_json(json.dumps({'response': json.dumps([{'name': 'Zoe'}])}))
    assert [n.name for n in args_dict['response']] == ['Zoe']

    # call_tool returns the inner typed value
    result = await toolset.call_tool('final_result', args_dict, _run_ctx(), tool)
    assert [n.name for n in result] == ['Zoe']
