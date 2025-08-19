from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from ..conftest import BinaryContent, try_import

with try_import() as imports_successful:
    from pydantic_ai.settings import ModelSettings
    from pydantic_evals.evaluators.llm_as_a_judge import (
        GradingOutput,
        _stringify,  # pyright: ignore[reportPrivateUsage]
        judge_input_output,
        judge_input_output_expected,
        judge_output,
        judge_output_expected,
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
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string output
    grading_output = await judge_output('Hello world', 'Content contains a greeting')
    assert isinstance(grading_output, GradingOutput)
    assert grading_output.reason == 'Test passed'
    assert grading_output.pass_ is True
    assert grading_output.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nContent contains a greeting\n</Rubric>' in call_args[0]


@pytest.mark.anyio
async def test_judge_output_with_model_settings_mock(mocker: MockerFixture):
    """Test judge_output function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    grading_output = await judge_output(
        'Hello world settings',
        'Content contains a greeting with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(grading_output, GradingOutput)
    assert grading_output.reason == 'Test passed with settings'
    assert grading_output.pass_ is True
    assert grading_output.score == 1.0

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert '<Output>\nHello world settings\n</Output>' in call_args[0]
    assert '<Rubric>\nContent contains a greeting with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs


@pytest.mark.anyio
async def test_judge_input_output_mock(mocker: MockerFixture):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

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


async def test_judge_input_output_binary_content_list_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    result = await judge_input_output([image_content, image_content], 'Hello world', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    raw_prompt = mock_run.call_args[0][0]

    # 1) It must be a list
    assert isinstance(raw_prompt, list), 'Expected prompt to be a list when passing binary'

    # 2) The BinaryContent you passed in should be one of the elements
    assert image_content in raw_prompt, 'Expected the exact BinaryContent instance to be in the prompt list'


async def test_judge_input_output_binary_content_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    result = await judge_input_output(image_content, 'Hello world', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    raw_prompt = mock_run.call_args[0][0]

    # 1) It must be a list
    assert isinstance(raw_prompt, list), 'Expected prompt to be a list when passing binary'

    # 2) The BinaryContent you passed in should be one of the elements
    assert image_content in raw_prompt, 'Expected the exact BinaryContent instance to be in the prompt list'


@pytest.mark.anyio
async def test_judge_input_output_with_model_settings_mock(mocker: MockerFixture):
    """Test judge_input_output function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    result = await judge_input_output(
        'Hello settings',
        'Hello world with settings',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert '<Input>\nHello settings\n</Input>' in call_args[0]
    assert '<Output>\nHello world with settings\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs


@pytest.mark.anyio
async def test_judge_input_output_expected_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_input_output_expected function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string input and output
    result = await judge_input_output_expected('Hello', 'Hello world', 'Hello', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    call_args = mock_run.call_args[0]
    assert '<Input>\nHello\n</Input>' in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input\n</Rubric>' in call_args[0]

    result = await judge_input_output_expected(image_content, 'Hello world', 'Hello', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args = mock_run.call_args[0]
    assert image_content in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input\n</Rubric>' in call_args[0]


@pytest.mark.anyio
async def test_judge_input_output_expected_with_model_settings_mock(
    mocker: MockerFixture, image_content: BinaryContent
):
    """Test judge_input_output_expected function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    result = await judge_input_output_expected(
        'Hello settings',
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args
    assert '<Input>\nHello settings\n</Input>' in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world with settings\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs

    result = await judge_input_output_expected(
        image_content,
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )

    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args
    assert image_content in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world with settings\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs

    result = await judge_input_output_expected(
        123,
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )

    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args

    assert call_args == snapshot(
        (
            [
                '<Input>\n',
                '123',
                '</Input>',
                """\
<Output>
Hello world with settings
</Output>\
""",
                """\
<Rubric>
Output contains input with settings
</Rubric>\
""",
                """\
<ExpectedOutput>
Hello
</ExpectedOutput>\
""",
            ],
        )
    )

    result = await judge_input_output_expected(
        [123],
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )

    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args

    assert call_args == snapshot(
        (
            [
                '<Input>\n',
                '123',
                '</Input>',
                """\
<Output>
Hello world with settings
</Output>\
""",
                """\
<Rubric>
Output contains input with settings
</Rubric>\
""",
                """\
<ExpectedOutput>
Hello
</ExpectedOutput>\
""",
            ],
        )
    )


@pytest.mark.anyio
async def test_judge_output_expected_mock(mocker: MockerFixture):
    """Test judge_output_expected function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string output and expected output
    result = await judge_output_expected('Hello world', 'Hello', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    call_args = mock_run.call_args[0]
    assert '<Input>' not in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input\n</Rubric>' in call_args[0]


@pytest.mark.anyio
async def test_judge_output_expected_with_model_settings_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_output_expected function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    result = await judge_output_expected(
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert '<Input>' not in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world with settings\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs

    result = await judge_output_expected(
        image_content,
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args
    assert '<Input>' not in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs
