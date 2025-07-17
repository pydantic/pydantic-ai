# Unit testing

Writing unit tests for Pydantic AI code is just like unit tests for any other Python code.

Because for the most part they're nothing new, we have pretty well established tools and patterns for writing and running these kinds of tests.

Unless you're really sure you know better, you'll probably want to follow roughly this strategy:

- Use [`pytest`](https://docs.pytest.org/en/stable/) as your test harness
- If you find yourself typing out long assertions, use [inline-snapshot](https://15r10nk.github.io/inline-snapshot/latest/)
- Similarly, [dirty-equals](https://dirty-equals.helpmanual.io/latest/) can be useful for comparing large data structures
- Use [`TestModel`][pydantic_ai.models.test.TestModel] or [`FunctionModel`][pydantic_ai.models.function.FunctionModel] in place of your actual model to avoid the usage, latency and variability of real LLM calls
- Use [`Agent.override`][pydantic_ai.agent.Agent.override] to replace an agent's model, dependencies, or toolsets inside your application logic
- Set [`ALLOW_MODEL_REQUESTS=False`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] globally to block any requests from being made to non-test models accidentally

### Unit testing with `TestModel`

The simplest and fastest way to exercise most of your application code is using [`TestModel`][pydantic_ai.models.test.TestModel], this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent.

!!! note "`TestModel` is not magic"
    The "clever" (but not too clever) part of `TestModel` is that it will attempt to generate valid structured data for [function tools](tools.md) and [output types](output.md#structured-output) based on the schema of the registered tools.

    There's no ML or AI in `TestModel`, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool.

    The resulting data won't look pretty or relevant, but it should pass Pydantic's validation in most cases.
    If you want something more sophisticated, use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] and write your own data generation logic.

Let's write unit tests for the following application code:

```python {title="weather_app.py"}
import asyncio
from datetime import date

from pydantic_ai import Agent, RunContext

from fake_database import DatabaseConn  # (1)!
from weather_service import WeatherService  # (2)!

weather_agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
def weather_forecast(
    ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():  # (3)!
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(  # (4)!
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts and save."""
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.output)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )
```

1. `DatabaseConn` is a class that holds a database connection
2. `WeatherService` has methods to get weather forecasts and historic data about the weather
3. We need to call a different endpoint depending on whether the date is in the past or the future, you'll see why this nuance is important below
4. This function is the code we want to test, together with the agent it uses

Here we have a function that takes a list of `#!python (user_prompt, user_id)` tuples, gets a weather forecast for each prompt, and stores the result in the database.

**We want to test this code without having to mock certain objects or modify our code so we can pass test objects in.**

Here's how we would write tests using [`TestModel`][pydantic_ai.models.test.TestModel]:

```python {title="test_weather_app.py" call_name="test_forecast" requires="weather_app.py"}
from datetime import timezone
import pytest

from dirty_equals import IsNow, IsStr

from pydantic_ai import models, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    ModelRequest,
)
from pydantic_ai.usage import Usage

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio  # (1)!
models.ALLOW_MODEL_REQUESTS = False  # (2)!


async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):  # (3)!
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)  # (4)!

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'  # (5)!

    assert messages == [  # (6)!
        ModelRequest(
            parts=[
                SystemPromptPart(
                    content='Providing a weather forecast at the locations the user provides.',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),  # (7)!
                ),
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',  # (8)!
                    },
                    tool_call_id=IsStr(),
                )
            ],
            usage=Usage(
                requests=1,
                request_tokens=71,
                response_tokens=7,
                total_tokens=78,
                details=None,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            usage=Usage(
                requests=1,
                request_tokens=77,
                response_tokens=16,
                total_tokens=93,
                details=None,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
        ),
    ]
```

1. We're using [anyio](https://anyio.readthedocs.io/en/stable/) to run async tests.
2. This is a safety measure to make sure we don't accidentally make real requests to the LLM while testing, see [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] for more details.
3. We're using [`Agent.override`][pydantic_ai.agent.Agent.override] to replace the agent's model with [`TestModel`][pydantic_ai.models.test.TestModel], the nice thing about `override` is that we can replace the model inside agent without needing access to the agent `run*` methods call site.
4. Now we call the function we want to test inside the `override` context manager.
5. But default, `TestModel` will return a JSON string summarising the tools calls made, and what was returned. If you wanted to customise the response to something more closely aligned with the domain, you could add [`custom_output_text='Sunny'`][pydantic_ai.models.test.TestModel.custom_output_text] when defining `TestModel`.
6. So far we don't actually know which tools were called and with which values, we can use [`capture_run_messages`][pydantic_ai.capture_run_messages] to inspect messages from the most recent run and assert the exchange between the agent and the model occurred as expected.
7. The [`IsNow`][dirty_equals.IsNow] helper allows us to use declarative asserts even with data which will contain timestamps that change over time.
8. `TestModel` isn't doing anything clever to extract values from the prompt, so these values are hardcoded.

### Unit testing with `FunctionModel`

The above tests are a great start, but careful readers will notice that the `WeatherService.get_forecast` is never called since `TestModel` calls `weather_forecast` with a date in the past.

To fully exercise `weather_forecast`, we need to use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] to customise how the tools is called.

Here's an example of using `FunctionModel` to test the `weather_forecast` tool with custom inputs

```python {title="test_weather_app2.py" call_name="test_forecast_future" requires="weather_app.py"}
import re

import pytest

from pydantic_ai import models
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


def call_weather_forecast(  # (1)!
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}  # (2)!
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])


async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):  # (3)!
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'
```

1. We define a function `call_weather_forecast` that will be called by `FunctionModel` in place of the LLM, this function has access to the list of [`ModelMessage`][pydantic_ai.messages.ModelMessage]s that make up the run, and [`AgentInfo`][pydantic_ai.models.function.AgentInfo] which contains information about the agent and the function tools and return tools.
2. Our function is slightly intelligent in that it tries to extract a date from the prompt, but just hard codes the location.
3. We use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] to replace the agent's model with our custom function.

### Overriding model via pytest fixtures

If you're writing lots of tests that all require model to be overridden, you can use [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) to override the model with [`TestModel`][pydantic_ai.models.test.TestModel] or [`FunctionModel`][pydantic_ai.models.function.FunctionModel] in a reusable way.

Here's an example of a fixture that overrides the model with `TestModel`:

```python {title="test_agent.py" requires="weather_app.py"}
import pytest
from weather_app import weather_agent

from pydantic_ai.models.test import TestModel


@pytest.fixture
def override_weather_agent():
    with weather_agent.override(model=TestModel()):
        yield


async def test_forecast(override_weather_agent: None):
    ...
    # test code here
```
