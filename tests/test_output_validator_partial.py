"""Tests for output validators with partial parameter support."""

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


def test_output_validator_with_partial_no_ctx():
    """Test output validator with partial parameter but no RunContext."""
    partial_calls = []
    final_calls = []

    m = TestModel(custom_output_text='test')
    agent = Agent(m)

    @agent.output_validator
    def validate_output(data: str, *, partial: bool) -> str:
        if partial:
            partial_calls.append(data)
            # During partial validation, allow incomplete data
            return data
        else:
            final_calls.append(data)
            # On final validation, apply transformation
            return data.upper()

    result = agent.run_sync('test')
    assert result.output == 'TEST'
    assert len(final_calls) == 1
    assert final_calls[0] == 'test'


def test_output_validator_with_partial_and_ctx():
    """Test output validator with both RunContext and partial parameter."""
    partial_calls = []
    final_calls = []

    m = TestModel(custom_output_text='test')
    agent = Agent(m, deps_type=str)

    @agent.output_validator
    def validate_output(ctx: RunContext[str], data: str, *, partial: bool) -> str:
        if partial:
            partial_calls.append((ctx.deps, data))
            return data
        else:
            final_calls.append((ctx.deps, data))
            return f'{ctx.deps}: {data}'

    result = agent.run_sync('test', deps='PREFIX')
    assert result.output == 'PREFIX: test'
    assert len(final_calls) == 1
    assert final_calls[0] == ('PREFIX', 'test')


def test_output_validator_without_partial_still_works():
    """Test that validators without partial parameter continue to work."""
    calls = []

    m = TestModel(custom_output_text='test')
    agent = Agent(m)

    @agent.output_validator
    def validate_output(data: str) -> str:
        calls.append(data)
        return data.upper()

    result = agent.run_sync('test')
    assert result.output == 'TEST'
    assert calls == ['test']


def test_output_validator_with_ctx_no_partial_still_works():
    """Test that validators with RunContext but no partial parameter continue to work."""
    calls = []

    m = TestModel(custom_output_text='test')
    agent = Agent(m, deps_type=str)

    @agent.output_validator
    def validate_output(ctx: RunContext[str], data: str) -> str:
        calls.append((ctx.deps, data))
        return f'{ctx.deps}: {data}'

    result = agent.run_sync('test', deps='PREFIX')
    assert result.output == 'PREFIX: test'
    assert calls == [('PREFIX', 'test')]


async def test_output_validator_streaming_with_partial():
    """Test output validator receives partial=True during streaming."""
    from pydantic_ai.models.test import TestModel

    partial_calls = []
    final_calls = []

    # Create a model that streams multiple chunks
    m = TestModel(seed=0)
    agent = Agent(m)

    @agent.output_validator
    def validate_output(data: str, *, partial: bool) -> str:
        if partial:
            partial_calls.append(data)
        else:
            final_calls.append(data)
        return data

    async with agent.run_stream('test') as run:
        chunks = []
        async for chunk in run.stream_text(delta=False):
            chunks.append(chunk)

    # During streaming, partial should be True
    assert len(partial_calls) > 0, 'Should have received partial validation calls during streaming'
    # Final result should have partial=False
    assert len(final_calls) == 1, 'Should have one final validation call'


async def test_output_validator_structured_with_partial():
    """Test output validator with structured output and partial parameter."""
    from pydantic import BaseModel

    partial_calls = []
    final_calls = []

    class OutputType(BaseModel):
        value: str

    m = TestModel()
    agent = Agent(m, output_type=OutputType)

    @agent.output_validator
    def validate_output(data: OutputType, *, partial: bool) -> OutputType:
        if partial:
            partial_calls.append(data)
        else:
            final_calls.append(data)
        return data

    async with agent.run_stream('test') as run:
        chunks = []
        async for chunk in run.stream_output():
            chunks.append(chunk)

    # Should receive both partial and final calls
    assert len(chunks) > 0
    # Final validation is called at least once (may be called multiple times during the stream completion)
    assert len(final_calls) >= 1


async def test_output_validator_partial_can_skip_validation():
    """Test that validators can conditionally skip checks during partial validation."""
    validation_count = {'partial': 0, 'final': 0}
    m = TestModel(custom_output_text='response')
    agent = Agent(m)

    @agent.output_validator
    def validate_output(data: str, *, partial: bool) -> str:
        # Track which validations are called
        if partial:
            validation_count['partial'] += 1
            # Skip expensive validation during partial
            return data
        else:
            validation_count['final'] += 1
            # Apply full validation on final result
            return data.upper()

    result = await agent.run('test')
    assert result.output == 'RESPONSE'
    assert validation_count['final'] == 1


def test_output_validator_signature_detection():
    """Test that the signature detection correctly identifies partial parameter."""
    from pydantic_ai._output import OutputValidator

    # Test with partial parameter
    def validator_with_partial(data: str, *, partial: bool) -> str:
        return data

    v1 = OutputValidator(validator_with_partial)
    assert v1._takes_partial is True  # pyright: ignore[reportPrivateUsage]
    assert v1._takes_ctx is False  # pyright: ignore[reportPrivateUsage]

    # Test without partial parameter
    def validator_without_partial(data: str) -> str:
        return data

    v2 = OutputValidator(validator_without_partial)
    assert v2._takes_partial is False  # pyright: ignore[reportPrivateUsage]
    assert v2._takes_ctx is False  # pyright: ignore[reportPrivateUsage]

    # Test with ctx and partial
    def validator_with_ctx_and_partial(ctx: RunContext[None], data: str, *, partial: bool) -> str:
        return data

    v3 = OutputValidator(validator_with_ctx_and_partial)
    assert v3._takes_partial is True  # pyright: ignore[reportPrivateUsage]
    assert v3._takes_ctx is True  # pyright: ignore[reportPrivateUsage]

    # Test with ctx but no partial
    def validator_with_ctx_no_partial(ctx: RunContext[None], data: str) -> str:
        return data

    v4 = OutputValidator(validator_with_ctx_no_partial)
    assert v4._takes_partial is False  # pyright: ignore[reportPrivateUsage]
    assert v4._takes_ctx is True  # pyright: ignore[reportPrivateUsage]


def test_output_validator_partial_must_be_keyword_only():
    """Test that partial parameter must be keyword-only."""
    from pydantic_ai._output import OutputValidator

    # Test that non-keyword-only partial raises ValueError
    def validator_with_positional_partial(data: str, partial: bool) -> str:
        return data

    with pytest.raises(ValueError, match='must be keyword-only'):
        OutputValidator(validator_with_positional_partial)
