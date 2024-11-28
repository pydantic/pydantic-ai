from black import timezonefrom black import timezonefrom black import timezone

# Testing and Evals

When thinking about PydanticAI use and LLM integrations in general, there are two distinct kinds of test:

1. **Unit tests** — tests of your application code, and whether it's behaving correctly
2. **"Evals"** — tests of the LLM, and how good or bad its responses are

For the most part, these two kinds of tests have pretty separate goals and considerations.

## Unit tests

Unit tests for PydanticAI code are just like unit tests for any other Python code.

Because for the most part they're nothing new, we have pretty well established tools and patterns for writing and running these kinds of tests.

Unless you're really sure you know better, you'll probably want to follow roughly this strategy:

* Use [`pytest`](https://docs.pytest.org/en/stable/) as your test harness
* If you find yourself typing out long assertions, use [`inline-snapshot`](https://15r10nk.github.io/inline-snapshot/latest/)
* Similarly, [dirty-equals](https://dirty-equals.helpmanual.io/latest/) can be useful for comparing large data structures
* Use [`TestModel`][pydantic_ai.models.test.TestModel] or [`FunctionModel`][pydantic_ai.models.function.FunctionModel] in place of your actual model to avoid the cost, latency and variability of real LLM calls
* Use [`Agent.override`][pydantic_ai.agent.Agent.override] to replace your model inside your application logic.
* Set [`ALLOW_MODEL_REQUESTS=False`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] globally to block any requests from being made to non-test models

### Unit testing with `TestModel`

The simplest and fastest way to exercise most of your application code is using [`TestModel`][pydantic_ai.models.test.TestModel], this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent.

!!! note "`TestModel` is not magic"
    The "clever" (but not too clever) part of `TestModel` is that it will attempt to generate valid structured data for [function tools](agents.md#function-tools) and [result types](results.md#structured-result-validation) based on the schema of the registered tools.

    There's no ML or AI in `TestModel`, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool.

    The resulting data won't look pretty or relevant, but it should pass Pydantic's validation in most cases.
    If you want something more sophisticated, use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] and write your own data generation logic.

Let's consider the following application code:

```py title="weather_app.py"
import asyncio
from datetime import date

from pydantic_ai import Agent, CallContext

from fake_database import DatabaseConn  # (1)!
from weather_service import WeatherService  # (2)!

weather_agent = Agent(
    'openai:gpt-4o',
    deps_type=WeatherService,
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
def weather_forecast(
    ctx: CallContext[WeatherService], location: str, forecast_date: date
) -> str:
    if forecast_date < date.today():
        # (pretend) use the cheaper endpoint to get historical data
        return ctx.deps.get_historic_weather(location, forecast_date)
    else:
        return ctx.deps.get_forecast(location, forecast_date)


async def run_weather_forecast(  # (3)!
    user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
    """Run weather forecast for a list of user prompts.

    Args:
        user_prompts: A list of tuples containing the user prompt and user id.
        conn: A database connection to store the forecast results.
    """
    async with WeatherService() as weather_service:

        async def run_forecast(prompt: str, user_id: int):
            result = await weather_agent.run(prompt, deps=weather_service)
            await conn.store_forecast(user_id, result.data)

        # run all prompts in parallel
        await asyncio.gather(
            *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
        )
```

1. `DatabaseConn` is a class that holds a database connection
2. `WeatherService` is a class that provides weather data
3. This function is the code we want to test, together with the agent it uses

Here we have a function that takes a list of `#!python (user_prompt, user_id)` tuples, gets a weather forecast for each prompt, and stores the result in the database.

We want to test this code without having to mock certain objects or modify our code so we can pass test objects in.

Here's how we would write tests using [`TestModel`][pydantic_ai.models.test.TestModel]:

```py title="test_weather_app.py"
from datetime import timezone
import pytest

from dirty_equals import IsNow

from pydantic_ai import models
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    SystemPrompt,
    UserPrompt,
    ModelStructuredResponse,
    ToolCall,
    ArgsObject,
    ToolReturn,
    ModelTextResponse,
)

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio  # (1)!
models.ALLOW_MODEL_REQUESTS = False  # (2)!


async def test_forecast_success():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=TestModel()):  # (3)!
        prompt = 'What will the weather be like in London on 2024-11-28?'
        await run_weather_forecast([(prompt, user_id)], conn)  # (4)!

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'  # (5)!

    assert weather_agent.last_run_messages == [  # (6)!
        SystemPrompt(
            content='Providing a weather forecast at the locations the user provides.',
            role='system',
        ),
        UserPrompt(
            content='What will the weather be like in London on 2024-11-28?',
            timestamp=IsNow(tz=timezone.utc),  # (7)!
            role='user',
        ),
        ModelStructuredResponse(
            calls=[
                ToolCall(
                    tool_name='weather_forecast',
                    args=ArgsObject(
                        args_object={'location': 'a', 'forecast_date': '2024-01-01'}
                    ),
                    tool_id=None,
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
            role='model-structured-response',
        ),
        ToolReturn(
            tool_name='weather_forecast',
            content='Sunny with a chance of rain',
            tool_id=None,
            timestamp=IsNow(tz=timezone.utc),
            role='tool-return',
        ),
        ModelTextResponse(
            content='{"weather_forecast":"Sunny with a chance of rain"}',
            timestamp=IsNow(tz=timezone.utc),
            role='model-text-response',
        ),
    ]
```

1. We're using [anyio](https://anyio.readthedocs.io/en/stable/) to run async tests.
2. This is a safety measure to make sure we don't accidentally make real requests to the LLM while testing.
3. We're using [`override`][pydantic_ai.agent.Agent.override] to replace the agent's model with [`TestModel`][pydantic_ai.models.test.TestModel].
4. Now we call the function we want to test.
5. But default, `TestModel` will return a JSON string summarising the tools calls made, and what was returned. If you wanted to customise the response to something more closely aligned with the domain, you could add [`custom_result_text='Sunny'`][pydantic_ai.models.test.TestModel.custom_result_text] when defining `TestModel`.
6. So far we don't actually know which tools were called and with which values, we can use the [`last_run_messages`][pydantic_ai.agent.Agent.last_run_messages] attribute to inspect messages from the most recent run and assert the exchange between the agent and the model occurred as expected.
7. The [`IsNow`][dirty_equals.IsNow] helper allows us to use declarative asserts even with data which will contain timestamps that change over time.

### Unit testing with `FunctionModel`

TODO

## Evals

TODO.

evals are more like benchmarks, they never "pass" although they do "fail", you care mostly about how they change over time, we (and we think most other people) don't really know what a "good" eval is, we provide some useful tools, we'll improve this if/when a common best practice emerges, or we think we have something interesting to say
