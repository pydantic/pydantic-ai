"""Tests for guardrails functionality."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from pydantic_ai import (
    Agent,
    GuardrailResult,
    InputGuardrail,
    InputGuardrailTripwireTriggered,
    ModelResponse,
    OutputGuardrail,
    OutputGuardrailTripwireTriggered,
    RunContext,
    TextPart,
)
from pydantic_ai.messages import ModelMessage, UserContent
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


# ============================================================
# Input Guardrail Tests
# ============================================================


async def test_input_guardrail_passes():
    """Test that passing guardrails don't block the agent."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    async def check_input(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.passed(message='Input is safe')

    result = await agent.run('Hello')
    assert result.output == 'Hello!'


async def test_input_guardrail_blocks():
    """Test that triggered guardrails raise InputGuardrailTripwireTriggered."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    async def block_harmful(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        if isinstance(prompt, str) and 'blocked' in prompt.lower():
            return GuardrailResult.blocked(message='Content contains blocked word')
        return GuardrailResult.passed()

    # This should pass
    result = await agent.run('Hello')
    assert result.output == 'Hello!'

    # This should be blocked
    with pytest.raises(InputGuardrailTripwireTriggered) as exc_info:
        await agent.run('This is blocked content')

    assert 'Content contains blocked word' in str(exc_info.value)
    assert exc_info.value.guardrail_name == 'block_harmful'
    assert exc_info.value.result.tripwire_triggered is True


async def test_input_guardrail_with_metadata():
    """Test that guardrails can return metadata."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    async def check_with_metadata(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        if isinstance(prompt, str) and 'secret' in prompt.lower():
            return GuardrailResult.blocked(
                message='PII detected',
                detected_pii_types=['secret'],
                severity='high',
            )
        return GuardrailResult.passed()

    with pytest.raises(InputGuardrailTripwireTriggered) as exc_info:
        await agent.run('This contains a secret')

    assert exc_info.value.result.metadata == {'detected_pii_types': ['secret'], 'severity': 'high'}


async def test_input_guardrail_blocking_mode():
    """Test that blocking guardrails run before non-blocking ones."""

    execution_order: list[str] = []

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail(blocking=True, name='blocking_guardrail')
    async def blocking_guardrail(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        execution_order.append('blocking')
        return GuardrailResult.passed()

    @agent.input_guardrail(blocking=False, name='non_blocking_guardrail')
    async def non_blocking_guardrail(
        ctx: RunContext[None], prompt: str | Sequence[UserContent]
    ) -> GuardrailResult[None]:
        execution_order.append('non_blocking')
        return GuardrailResult.passed()

    await agent.run('Hello')

    # Blocking guardrail should run before non-blocking
    assert execution_order == ['blocking', 'non_blocking']


async def test_multiple_input_guardrails():
    """Test that multiple guardrails all run."""

    guardrails_run: list[str] = []

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    async def guardrail_1(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        guardrails_run.append('guardrail_1')
        return GuardrailResult.passed()

    @agent.input_guardrail
    async def guardrail_2(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        guardrails_run.append('guardrail_2')
        return GuardrailResult.passed()

    @agent.input_guardrail
    async def guardrail_3(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        guardrails_run.append('guardrail_3')
        return GuardrailResult.passed()

    await agent.run('Hello')
    assert len(guardrails_run) == 3


async def test_input_guardrail_first_failure_stops():
    """Test that the first guardrail failure raises exception immediately."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail(blocking=True, name='first')
    async def first_guardrail(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.blocked(message='First guardrail blocked')

    @agent.input_guardrail(blocking=True, name='second')
    async def second_guardrail(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        # This should not run since the first one blocked
        raise AssertionError('Second guardrail should not have run')

    with pytest.raises(InputGuardrailTripwireTriggered) as exc_info:
        await agent.run('Hello')

    assert exc_info.value.guardrail_name == 'first'


async def test_input_guardrail_with_deps():
    """Test that guardrails can access agent dependencies."""

    @dataclass
    class MyDeps:
        blocked_words: list[str]

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent: Agent[MyDeps, str] = Agent(FunctionModel(simple_model), deps_type=MyDeps)

    @agent.input_guardrail
    async def check_blocked_words(
        ctx: RunContext[MyDeps], prompt: str | Sequence[UserContent]
    ) -> GuardrailResult[None]:
        if isinstance(prompt, str):
            for word in ctx.deps.blocked_words:
                if word in prompt.lower():
                    return GuardrailResult.blocked(message=f'Word "{word}" is blocked')
        return GuardrailResult.passed()

    # Pass with no blocked words
    deps = MyDeps(blocked_words=['spam', 'malware'])
    result = await agent.run('Hello world', deps=deps)
    assert result.output == 'Hello!'

    # Block with blocked word
    with pytest.raises(InputGuardrailTripwireTriggered):
        await agent.run('This is spam content', deps=deps)


async def test_input_guardrail_sync_function():
    """Test that sync guardrail functions work correctly."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    def sync_guardrail(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        if isinstance(prompt, str) and 'blocked' in prompt.lower():
            return GuardrailResult.blocked(message='Blocked by sync guardrail')
        return GuardrailResult.passed()

    result = await agent.run('Hello')
    assert result.output == 'Hello!'

    with pytest.raises(InputGuardrailTripwireTriggered):
        await agent.run('This is blocked')


# ============================================================
# Output Guardrail Tests
# ============================================================


async def test_output_guardrail_passes():
    """Test that passing output guardrails don't block the response."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Safe response')])

    agent = Agent(FunctionModel(simple_model))

    @agent.output_guardrail
    async def check_output(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        return GuardrailResult.passed()

    result = await agent.run('Hello')
    assert result.output == 'Safe response'


async def test_output_guardrail_blocks():
    """Test that triggered output guardrails raise OutputGuardrailTripwireTriggered."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('This contains SECRET data')])

    agent = Agent(FunctionModel(simple_model))

    @agent.output_guardrail
    async def block_secrets(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        if 'SECRET' in output:
            return GuardrailResult.blocked(message='Output contains secrets')
        return GuardrailResult.passed()

    with pytest.raises(OutputGuardrailTripwireTriggered) as exc_info:
        await agent.run('Tell me a secret')

    assert 'Output contains secrets' in str(exc_info.value)
    assert exc_info.value.guardrail_name == 'block_secrets'


async def test_output_guardrail_with_structured_output():
    """Test output guardrails with typed output."""
    from pydantic import BaseModel

    class Response(BaseModel):
        message: str
        confidence: float

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[TextPart('{"message": "High confidence response", "confidence": 0.95}')])

    agent = Agent(FunctionModel(simple_model), output_type=Response)

    @agent.output_guardrail
    async def check_confidence(ctx: RunContext[None], output: Response) -> GuardrailResult[None]:
        if output.confidence < 0.5:
            return GuardrailResult.blocked(message='Confidence too low')
        return GuardrailResult.passed()

    result = await agent.run('Hello')
    assert result.output.message == 'High confidence response'
    assert result.output.confidence == 0.95


async def test_output_guardrail_with_structured_output_blocked():
    """Test output guardrails blocking typed output."""
    from pydantic import BaseModel

    class Response(BaseModel):
        message: str
        confidence: float

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[TextPart('{"message": "Low confidence response", "confidence": 0.3}')])

    agent = Agent(FunctionModel(simple_model), output_type=Response)

    @agent.output_guardrail
    async def check_confidence(ctx: RunContext[None], output: Response) -> GuardrailResult[None]:
        if output.confidence < 0.5:
            return GuardrailResult.blocked(message='Confidence too low')
        return GuardrailResult.passed()

    with pytest.raises(OutputGuardrailTripwireTriggered) as exc_info:
        await agent.run('Hello')

    assert 'Confidence too low' in str(exc_info.value)


async def test_multiple_output_guardrails():
    """Test that multiple output guardrails all run."""

    guardrails_run: list[str] = []

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.output_guardrail
    async def output_guardrail_1(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        guardrails_run.append('output_guardrail_1')
        return GuardrailResult.passed()

    @agent.output_guardrail
    async def output_guardrail_2(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        guardrails_run.append('output_guardrail_2')
        return GuardrailResult.passed()

    await agent.run('Hello')
    assert guardrails_run == ['output_guardrail_1', 'output_guardrail_2']


async def test_output_guardrail_sync_function():
    """Test that sync output guardrail functions work correctly."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.output_guardrail
    def sync_output_guardrail(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        if 'blocked' in output.lower():
            return GuardrailResult.blocked(message='Output contains blocked content')
        return GuardrailResult.passed()

    result = await agent.run('Hello')
    assert result.output == 'Hello!'


# ============================================================
# Combined Input and Output Guardrail Tests
# ============================================================


async def test_both_guardrails():
    """Test that both input and output guardrails run."""

    guardrails_run: list[str] = []

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Response')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    async def input_check(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        guardrails_run.append('input_guardrail')
        return GuardrailResult.passed()

    @agent.output_guardrail
    async def output_check(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        guardrails_run.append('output_guardrail')
        return GuardrailResult.passed()

    await agent.run('Hello')
    assert guardrails_run == ['input_guardrail', 'output_guardrail']


async def test_input_blocked_skips_agent():
    """Test that if input guardrail blocks, the agent doesn't run."""

    agent_called = False

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal agent_called
        agent_called = True
        return ModelResponse(parts=[TextPart('Response')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    async def block_all(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.blocked(message='All input blocked')

    with pytest.raises(InputGuardrailTripwireTriggered):
        await agent.run('Hello')

    assert not agent_called, 'Agent should not have been called when input was blocked'


# ============================================================
# GuardrailResult Tests
# ============================================================


def test_guardrail_result_passed():
    """Test GuardrailResult.passed() factory method."""
    result = GuardrailResult.passed()
    assert result.tripwire_triggered is False
    assert result.output is None
    assert result.message is None
    assert result.metadata == {}


def test_guardrail_result_passed_with_output():
    """Test GuardrailResult.passed() with output."""

    @dataclass
    class ClassificationOutput:
        category: str
        score: float

    output = ClassificationOutput(category='safe', score=0.99)
    # Type narrowing via explicit construction to test runtime behavior
    result = GuardrailResult(tripwire_triggered=False, output=output, message='Classification complete')

    assert result.tripwire_triggered is False
    assert result.output == output
    assert result.message == 'Classification complete'


def test_guardrail_result_blocked():
    """Test GuardrailResult.blocked() factory method."""
    result = GuardrailResult.blocked(message='Content blocked')
    assert result.tripwire_triggered is True
    assert result.message == 'Content blocked'


def test_guardrail_result_blocked_with_metadata():
    """Test GuardrailResult.blocked() with metadata kwargs."""
    result = GuardrailResult.blocked(
        message='PII detected',
        detected_types=['email', 'phone'],
        risk_level='high',
    )

    assert result.tripwire_triggered is True
    assert result.message == 'PII detected'
    assert result.metadata == {'detected_types': ['email', 'phone'], 'risk_level': 'high'}


# ============================================================
# Constructor-based Guardrail Tests
# ============================================================


async def test_guardrails_via_constructor():
    """Test adding guardrails via constructor instead of decorators."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    async def input_guardrail_func(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        if isinstance(prompt, str) and 'blocked' in prompt.lower():
            return GuardrailResult.blocked(message='Input blocked')
        return GuardrailResult.passed()

    async def output_guardrail_func(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        if 'secret' in output.lower():
            return GuardrailResult.blocked(message='Output contains secrets')
        return GuardrailResult.passed()

    agent = Agent(
        FunctionModel(simple_model),
        input_guardrails=[InputGuardrail(function=input_guardrail_func)],
        output_guardrails=[OutputGuardrail(function=output_guardrail_func)],
    )

    # Test passing input
    result = await agent.run('Hello')
    assert result.output == 'Hello!'

    # Test blocked input
    with pytest.raises(InputGuardrailTripwireTriggered):
        await agent.run('This is blocked')


# ============================================================
# Exception Tests
# ============================================================


def test_input_guardrail_tripwire_exception():
    """Test InputGuardrailTripwireTriggered exception attributes."""
    result = GuardrailResult.blocked(message='Test message')
    exc = InputGuardrailTripwireTriggered('test_guardrail', result)

    assert exc.guardrail_name == 'test_guardrail'
    assert exc.result == result
    assert 'Test message' in str(exc)


def test_output_guardrail_tripwire_exception():
    """Test OutputGuardrailTripwireTriggered exception attributes."""
    result = GuardrailResult.blocked(message='Test output message')
    exc = OutputGuardrailTripwireTriggered('test_output_guardrail', result)

    assert exc.guardrail_name == 'test_output_guardrail'
    assert exc.result == result
    assert 'Test output message' in str(exc)


def test_guardrail_exception_default_message():
    """Test that exceptions have default messages when result.message is None."""
    result = GuardrailResult.blocked()  # No message

    input_exc = InputGuardrailTripwireTriggered('my_guardrail', result)
    assert 'my_guardrail' in str(input_exc)

    output_exc = OutputGuardrailTripwireTriggered('my_output_guardrail', result)
    assert 'my_output_guardrail' in str(output_exc)


def test_guardrail_exception_inherits_from_agent_run_error():
    """Test that guardrail exceptions inherit from AgentRunError."""
    from pydantic_ai.exceptions import AgentRunError

    result = GuardrailResult.blocked(message='Test')
    input_exc = InputGuardrailTripwireTriggered('test', result)
    output_exc = OutputGuardrailTripwireTriggered('test', result)

    assert isinstance(input_exc, AgentRunError)
    assert isinstance(output_exc, AgentRunError)


# ============================================================
# Run Sync Tests
# ============================================================


def test_guardrails_with_run_sync():
    """Test that guardrails work with run_sync."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    def sync_input_guardrail(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.passed()

    @agent.output_guardrail
    def sync_output_guardrail(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        return GuardrailResult.passed()

    result = agent.run_sync('Hello')
    assert result.output == 'Hello!'


def test_guardrails_blocking_with_run_sync():
    """Test that guardrails can block with run_sync."""

    def simple_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Hello!')])

    agent = Agent(FunctionModel(simple_model))

    @agent.input_guardrail
    def block_all(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.blocked(message='All blocked')

    with pytest.raises(InputGuardrailTripwireTriggered):
        agent.run_sync('Hello')


# ============================================================
# Streaming Tests
# ============================================================


async def test_guardrails_with_streaming():
    """Test that guardrails work with run_stream."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())

    @agent.input_guardrail
    async def check_input(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.passed()

    @agent.output_guardrail
    async def check_output(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        return GuardrailResult.passed()

    async with agent.run_stream('Hello') as result:
        output = await result.get_output()
        assert isinstance(output, str)


async def test_input_guardrail_blocks_streaming():
    """Test that input guardrails can block streaming."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())

    @agent.input_guardrail
    async def block_all(ctx: RunContext[None], prompt: str | Sequence[UserContent]) -> GuardrailResult[None]:
        return GuardrailResult.blocked(message='Blocked')

    with pytest.raises(InputGuardrailTripwireTriggered):
        async with agent.run_stream('Hello'):
            pass


async def test_output_guardrail_passes_streaming():
    """Test that passing output guardrails work with streaming."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(custom_output_text='Safe data'))

    @agent.output_guardrail
    async def check_output(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
        # Output guardrails run after streaming result is finalized
        return GuardrailResult.passed()

    async with agent.run_stream('Hello') as result:
        output = await result.get_output()
        assert output == 'Safe data'
