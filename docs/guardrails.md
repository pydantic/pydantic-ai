# Guardrails

Guardrails provide a way to validate, filter, and control agent inputs and outputs. They help you enforce safety policies, compliance requirements, or custom business rules before and after your agent processes requests.

## Overview

Pydantic AI supports two types of guardrails:

- **Input guardrails**: Validate user prompts _before_ the agent runs, allowing you to reject harmful or off-topic inputs without consuming tokens
- **Output guardrails**: Validate agent responses _before_ returning them to the user, allowing you to catch and handle problematic outputs

Both types use a simple decorator-based API consistent with other Pydantic AI patterns like `@agent.tool` and `@agent.system_prompt`.

## Basic Usage

### Input Guardrails

Input guardrails validate user prompts before the agent processes them:

```python {title="input_guardrail_example.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent

agent = Agent('openai:gpt-4o')


@agent.input_guardrail
async def check_input(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    if isinstance(prompt, str) and 'blocked_word' in prompt.lower():
        return GuardrailResult.blocked(message='Content contains blocked word')
    return GuardrailResult.passed()
```

### Output Guardrails

Output guardrails validate agent responses before they are returned:

```python {title="output_guardrail_example.py"}
from pydantic_ai import Agent, GuardrailResult, RunContext

agent = Agent('openai:gpt-4o')


@agent.output_guardrail
async def check_output(
    ctx: RunContext[None], agent: Agent[None, str], output: str
) -> GuardrailResult[None]:
    if 'SECRET' in output:
        return GuardrailResult.blocked(message='Output contains sensitive information')
    return GuardrailResult.passed()
```

## GuardrailResult

The [`GuardrailResult`][pydantic_ai.guardrails.GuardrailResult] class represents the outcome of a guardrail check. It provides factory methods for common cases:

### Passing

Use `GuardrailResult.passed()` when the content is acceptable:

```python {title="guardrail_passed.py"}
from pydantic_ai import GuardrailResult

# Simple pass
result = GuardrailResult.passed()

# Pass with a message
result = GuardrailResult.passed(message='Content is safe')

# Pass with structured output (useful for classification guardrails)
result = GuardrailResult.passed(output={'category': 'safe', 'confidence': 0.98})
```

### Blocking

Use `GuardrailResult.blocked()` when the content should be rejected:

```python {title="guardrail_blocked.py"}
from pydantic_ai import GuardrailResult

# Simple block
result = GuardrailResult.blocked(message='Content blocked')

# Block with metadata for logging/auditing
result = GuardrailResult.blocked(
    message='PII detected in input',
    detected_pii_types=['email', 'phone'],
    risk_level='high',
)
```

Any keyword arguments passed to `blocked()` are stored in the `metadata` field for logging and debugging.

## Handling Blocked Content

When a guardrail triggers (returns `GuardrailResult.blocked()`), an exception is raised:

- [`InputGuardrailTripwireTriggered`][pydantic_ai.guardrails.InputGuardrailTripwireTriggered] for input guardrails
- [`OutputGuardrailTripwireTriggered`][pydantic_ai.guardrails.OutputGuardrailTripwireTriggered] for output guardrails

```python {title="handling_blocked.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.guardrails import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)

agent = Agent('openai:gpt-4o')
# ... register guardrails ...

try:
    result = await agent.run('potentially harmful content')
except InputGuardrailTripwireTriggered as e:
    print(f'Input blocked by {e.guardrail_name}: {e.result.message}')
    print(f'Metadata: {e.result.metadata}')
except OutputGuardrailTripwireTriggered as e:
    print(f'Output blocked by {e.guardrail_name}: {e.result.message}')
```

## Execution Modes

### Parallel vs Blocking Input Guardrails

Input guardrails can be configured to run in two modes:

- **Blocking mode** (`run_in_parallel=False`): Guardrails run sequentially before the agent starts. If any blocking guardrail fails, the agent never runs.
- **Parallel mode** (`run_in_parallel=True`, default): Multiple guardrails run concurrently with each other.

Note: All input guardrails run before the model request is made. The `run_in_parallel` setting controls whether multiple guardrails run concurrently with each other, not whether they run during agent execution.

```python {title="blocking_guardrail.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent

agent = Agent('openai:gpt-4o')


@agent.input_guardrail(run_in_parallel=False)
async def must_pass_first(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    """This guardrail must pass before the agent starts running."""
    # Expensive check that should complete before model invocation
    return GuardrailResult.passed()


@agent.input_guardrail(run_in_parallel=True)  # This is the default
async def can_run_in_parallel(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    """This guardrail runs concurrently with other parallel guardrails."""
    return GuardrailResult.passed()
```

**Execution order:**

1. All blocking guardrails (`run_in_parallel=False`) run first, sequentially
2. If any blocking guardrail triggers, execution stops immediately
3. All parallel guardrails (`run_in_parallel=True`) run concurrently with each other
4. If any parallel guardrail triggers, an exception is raised
5. Only after all guardrails pass does the model request begin

### Output Guardrail Execution

Output guardrails always run sequentially after the agent completes, in the order they were registered.

## Using Dependencies

Guardrails have full access to the agent's dependencies through [`RunContext`][pydantic_ai.RunContext]:

```python {title="guardrail_with_deps.py" test="skip"}
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent


@dataclass
class ContentModerationDeps:
    blocked_words: list[str]
    allowed_topics: list[str]


agent: Agent[ContentModerationDeps, str] = Agent(
    'openai:gpt-4o',
    deps_type=ContentModerationDeps,
)


@agent.input_guardrail
async def check_blocked_words(
    ctx: RunContext[ContentModerationDeps],
    agent: Agent[ContentModerationDeps, str],
    prompt: str | Sequence[UserContent],
) -> GuardrailResult[None]:
    if isinstance(prompt, str):
        for word in ctx.deps.blocked_words:
            if word.lower() in prompt.lower():
                return GuardrailResult.blocked(
                    message=f'Content contains blocked word: {word}',
                    blocked_word=word,
                )
    return GuardrailResult.passed()


# Usage
deps = ContentModerationDeps(
    blocked_words=['spam', 'scam'],
    allowed_topics=['customer support', 'product info'],
)
result = await agent.run('Hello, I need help', deps=deps)
```

## Structured Output Guardrails

Output guardrails receive the validated output type, making them perfect for validating structured responses:

```python {title="structured_output_guardrail.py" test="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, GuardrailResult, RunContext


class Response(BaseModel):
    answer: str
    confidence: float


agent: Agent[None, Response] = Agent(
    'openai:gpt-4o',
    output_type=Response,
)


@agent.output_guardrail
async def check_confidence(
    ctx: RunContext[None], agent: Agent[None, Response], output: Response
) -> GuardrailResult[None]:
    if output.confidence < 0.5:
        return GuardrailResult.blocked(
            message='Response confidence too low',
            confidence=output.confidence,
        )
    return GuardrailResult.passed()
```

## Sync and Async Functions

Guardrails can be either sync or async functions. Sync functions are automatically run in an executor:

```python {title="sync_guardrail.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent

agent = Agent('openai:gpt-4o')


@agent.input_guardrail
def sync_guardrail(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    # Sync function - runs in executor
    return GuardrailResult.passed()


@agent.input_guardrail
async def async_guardrail(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    # Async function - runs directly
    return GuardrailResult.passed()
```

## Constructor-Based Registration

You can also register guardrails via the Agent constructor using [`InputGuardrail`][pydantic_ai._guardrail.InputGuardrail] and [`OutputGuardrail`][pydantic_ai._guardrail.OutputGuardrail]:

```python {title="constructor_guardrails.py" test="skip"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai._guardrail import InputGuardrail, OutputGuardrail
from pydantic_ai.messages import UserContent


async def my_input_guardrail(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    return GuardrailResult.passed()


async def my_output_guardrail(
    ctx: RunContext[None], agent: Agent[None, str], output: str
) -> GuardrailResult[None]:
    return GuardrailResult.passed()


agent = Agent(
    'openai:gpt-4o',
    input_guardrails=[
        InputGuardrail(function=my_input_guardrail, run_in_parallel=True),
    ],
    output_guardrails=[
        OutputGuardrail(function=my_output_guardrail),
    ],
)
```

## Multiple Guardrails

You can register multiple guardrails of each type. They are executed in registration order:

```python {title="multiple_guardrails.py" test="skip"}
from collections.abc import Sequence

from pydantic_ai import Agent, GuardrailResult, RunContext
from pydantic_ai.messages import UserContent

agent = Agent('openai:gpt-4o')


@agent.input_guardrail(name='profanity_filter')
async def check_profanity(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    # Check for profanity
    return GuardrailResult.passed()


@agent.input_guardrail(name='pii_detector')
async def detect_pii(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    # Detect personally identifiable information
    return GuardrailResult.passed()


@agent.input_guardrail(name='topic_classifier')
async def classify_topic(
    ctx: RunContext[None], agent: Agent[None, str], prompt: str | Sequence[UserContent]
) -> GuardrailResult[None]:
    # Ensure the prompt is on-topic
    return GuardrailResult.passed()
```

## Best Practices

1. **Use blocking mode for critical checks**: If a guardrail must pass before the agent starts (e.g., authentication), use `run_in_parallel=False`.

2. **Keep guardrails fast**: Guardrails add latency. Use async functions and avoid blocking I/O.

3. **Include metadata for debugging**: Use the metadata kwargs in `GuardrailResult.blocked()` to include useful debugging information.

4. **Name your guardrails**: Use the `name` parameter for clearer error messages and logging.

5. **Handle exceptions gracefully**: Always catch guardrail exceptions in your application code and provide appropriate user feedback.

6. **Consider using dependencies**: Pass configuration through dependencies rather than hardcoding values.

## API Reference

- [`GuardrailResult`][pydantic_ai.guardrails.GuardrailResult]
- [`InputGuardrailTripwireTriggered`][pydantic_ai.guardrails.InputGuardrailTripwireTriggered]
- [`OutputGuardrailTripwireTriggered`][pydantic_ai.guardrails.OutputGuardrailTripwireTriggered]
- [`InputGuardrail`][pydantic_ai._guardrail.InputGuardrail]
- [`OutputGuardrail`][pydantic_ai._guardrail.OutputGuardrail]
