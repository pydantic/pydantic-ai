# `pydantic_ai.models.function`

A model controlled by a local function.

[`FunctionModel`][pydantic_ai.models.function.FunctionModel] is similar to [`TestModel`](test.md),
but allows greater control over the model's behavior.

Its primary use case is for more advanced unit testing than is possible with `TestModel`.

Here's a minimal example:

```py {title="function_model_usage.py" call_name="test_my_agent" noqa="I001"}
from pydantic_ai import Agent
from pydantic_ai import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel, AgentInfo

my_agent = Agent('openai:gpt-5.2')


async def model_function(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    print(messages)
    """
    [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='Testing my agent...',
                    timestamp=datetime.datetime(...),
                )
            ],
            timestamp=datetime.datetime(...),
            run_id='...',
        )
    ]
    """
    print(info)
    """
    AgentInfo(
        function_tools=[],
        allow_text_output=True,
        output_tools=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[], builtin_tools=[], output_tools=[]
        ),
        instructions=None,
    )
    """
    return ModelResponse(parts=[TextPart('hello world')])


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    with my_agent.override(model=FunctionModel(model_function)):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'hello world'
```

See [Unit testing with `FunctionModel`](../../testing.md#unit-testing-with-functionmodel) for detailed documentation.

::: pydantic_ai.models.function
