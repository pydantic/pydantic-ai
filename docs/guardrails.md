# Guardrails

Guardrails validate agent inputs and outputs before and after execution.

## Basic Usage

### Input Guardrails

```python {title="input_guardrail_example.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent

agent = Agent('openai:gpt-4o')


@agent.input_guardrail
async def check_input(
    ctx: RunContext[None], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    if isinstance(prompt, str) and 'blocked' in prompt.lower():
        return GuardrailResult.blocked(message='Content blocked')
    return GuardrailResult.passed()
```

### Output Guardrails

```python {title="output_guardrail_example.py"}
from pydantic_ai import Agent, GuardrailResult, RunContext

agent = Agent('openai:gpt-4o')


@agent.output_guardrail
async def check_output(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
    if 'SECRET' in output:
        return GuardrailResult.blocked(message='Output contains sensitive information')
    return GuardrailResult.passed()
```

## GuardrailResult

[`GuardrailResult`][pydantic_ai.guardrails.GuardrailResult] indicates whether a guardrail passed or blocked:

```python {title="guardrail_result.py"}
from pydantic_ai import GuardrailResult

# Passed
result = GuardrailResult.passed()
result = GuardrailResult.passed(message='Content is safe')

# Blocked with metadata
result = GuardrailResult.blocked(
    message='PII detected',
    detected_types=['email', 'phone'],
)
```

## Handling Blocked Content

Blocked guardrails raise exceptions:

```python {title="handling_blocked.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.guardrails import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

agent = Agent('openai:gpt-4o')


async def main():
    try:
        result = await agent.run('potentially harmful content')
    except InputGuardrailTripwireTriggered as e:
        print(f'Input blocked by {e.guardrail_name}: {e.result.message}')
    except OutputGuardrailTripwireTriggered as e:
        print(f'Output blocked by {e.guardrail_name}: {e.result.message}')
```

## Blocking vs Non-blocking Guardrails

By default, input guardrails run concurrently with each other. Use `blocking=True` to run a guardrail before others:

```python {title="blocking_guardrail.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent

agent = Agent('openai:gpt-4o')


@agent.input_guardrail(blocking=True)
async def must_pass_first(
    ctx: RunContext[None], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    """This guardrail must pass before other guardrails run."""
    return GuardrailResult.passed()


@agent.input_guardrail  # blocking=False is the default
async def can_run_concurrently(
    ctx: RunContext[None], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    """Runs concurrently with other non-blocking guardrails."""
    return GuardrailResult.passed()
```

**Execution order:**

1. All blocking guardrails run first, sequentially
2. If any blocking guardrail triggers, execution stops
3. All non-blocking guardrails run concurrently
4. If any guardrail triggers, an exception is raised
5. Only after all guardrails pass does the model request begin

## Using Dependencies

Guardrails have access to dependencies via [`RunContext`][pydantic_ai.RunContext]:

```python {title="guardrail_with_deps.py" test="skip"}
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent


@dataclass
class Deps:
    blocked_words: list[str]


agent: Agent[Deps, str] = Agent('openai:gpt-4o', deps_type=Deps)


@agent.input_guardrail
async def check_blocked_words(
    ctx: RunContext[Deps], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    if isinstance(prompt, str):
        for word in ctx.deps.blocked_words:
            if word in prompt.lower():
                return GuardrailResult.blocked(message=f'Blocked word: {word}')
    return GuardrailResult.passed()
```

## Structured Output Guardrails

Output guardrails receive the validated output type:

```python {title="structured_output_guardrail.py" test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, GuardrailResult, RunContext


class Response(BaseModel):
    answer: str
    confidence: float


agent: Agent[None, Response] = Agent('openai:gpt-4o', output_type=Response)


@agent.output_guardrail
async def check_confidence(ctx: RunContext[None], output: Response) -> GuardrailResult[None]:
    if output.confidence < 0.5:
        return GuardrailResult.blocked(message='Confidence too low')
    return GuardrailResult.passed()
```

## Constructor-Based Registration

Register guardrails via the constructor using [`InputGuardrail`][pydantic_ai.guardrails.InputGuardrail] and [`OutputGuardrail`][pydantic_ai.guardrails.OutputGuardrail]:

```python {title="constructor_guardrails.py" test="skip"}
from collections.abc import Sequence

from pydantic_ai import (
    Agent,
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    RunContext,
)
from pydantic_ai.messages import UserContent


async def my_input_guardrail(
    ctx: RunContext[None], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    return GuardrailResult.passed()


async def my_output_guardrail(ctx: RunContext[None], output: str) -> GuardrailResult[None]:
    return GuardrailResult.passed()


agent = Agent(
    'openai:gpt-4o',
    input_guardrails=[InputGuardrail(function=my_input_guardrail)],
    output_guardrails=[OutputGuardrail(function=my_output_guardrail)],
)
```

## API Reference

- [`GuardrailResult`][pydantic_ai.guardrails.GuardrailResult]
- [`InputGuardrailTripwireTriggered`][pydantic_ai.guardrails.InputGuardrailTripwireTriggered]
- [`OutputGuardrailTripwireTriggered`][pydantic_ai.guardrails.OutputGuardrailTripwireTriggered]
- [`InputGuardrail`][pydantic_ai.guardrails.InputGuardrail]
- [`OutputGuardrail`][pydantic_ai.guardrails.OutputGuardrail]
