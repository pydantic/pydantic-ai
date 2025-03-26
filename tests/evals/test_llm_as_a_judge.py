from __future__ import annotations as _annotations

import pytest
from pytest_mock import MockerFixture

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators.llm_as_a_judge import (
        GradingOutput,
        _stringify,  # pyright: ignore[reportPrivateUsage]
        judge_input_output,
        judge_output,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


def test_grading_output():
    """Test GradingOutput model."""
    # Test with pass=True
    output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    assert output.reason == 'Test passed'
    assert output.pass_ is True
    assert output.score == 1.0

    # Test with pass=False
    output = GradingOutput(reason='Test failed', pass_=False, score=0.0)
    assert output.reason == 'Test failed'
    assert output.pass_ is False
    assert output.score == 0.0

    # Test with alias
    output = GradingOutput.model_validate({'reason': 'Test passed', 'pass': True, 'score': 1.0})
    assert output.reason == 'Test passed'
    assert output.pass_ is True
    assert output.score == 1.0


def test_stringify():
    """Test _stringify function."""
    # Test with string
    assert _stringify('test') == 'test'

    # Test with dict
    assert _stringify({'key': 'value'}) == '{"key":"value"}'

    # Test with list
    assert _stringify([1, 2, 3]) == '[1,2,3]'

    # Test with custom object
    class CustomObject:
        def __repr__(self):
            return 'CustomObject()'

    obj = CustomObject()
    assert _stringify(obj) == 'CustomObject()'

    # Test with non-JSON-serializable object
    class NonSerializable:
        def __repr__(self):
            return 'NonSerializable()'

    obj = NonSerializable()
    assert _stringify(obj) == 'NonSerializable()'


@pytest.mark.anyio
async def test_judge_output_mock(mocker: MockerFixture):
    """Test judge_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.data = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.Agent.run', return_value=mock_result)

    # Test with string output
    result = await judge_output('Hello world', 'Content contains a greeting')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nContent contains a greeting\n</Rubric>' in call_args[0]


@pytest.mark.anyio
async def test_judge_input_output_mock(mocker: MockerFixture):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.data = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.Agent.run', return_value=mock_result)

    # Test with string input and output
    result = await judge_input_output('Hello', 'Hello world', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0]
    assert '<Input>\nHello\n</Input>' in call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input\n</Rubric>' in call_args[0]
