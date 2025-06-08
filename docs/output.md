"Output" refers to the final value returned from [running an agent](agents.md#running-agents). This can be either plain text, [structured data](#structured-output), or the result of a [function](#output-functions) called with arguments provided by the model.

The output is wrapped in [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] or [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] so that you can access other data, like [usage][pydantic_ai.usage.Usage] of the run and [message history](message-history.md#accessing-messages-from-results).

Both `AgentRunResult` and `StreamedRunResult` are generic in the data they wrap, so typing information about the data returned by the agent is preserved.

A run ends when a plain text response is received (assuming no output type is specified or `str` is one of the allowed options), or when the model responds with one of the structured output types by calling a special output tool. A run can also be cancelled if usage limits are exceeded, see [Usage Limits](agents.md#usage-limits).

Here's an example using a Pydantic model as the `output_type`, forcing the model to respond with data matching our specification:

```python {title="olympics.py" line_length="90"}
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-1.5-flash', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

_(This example is complete, it can be run "as is")_

## Output data {#structured-output}

The [`Agent`][pydantic_ai.Agent] class constructor takes an `output_type` argument that takes one or more types or [output functions](#output-functions). It supports both type unions and lists of types and functions.

When no output type is specified, or when the output type is `str` or a union or list of types including `str`, the model is allowed to respond with plain text, and this text is used as the output data.
If `str` is not among the allowed output types, the model is not allowed to respond with plain text and is forced to return structured data (or arguments to an output function).

If the output type is a union or list with multiple members, each member (except for `str`, if it is a member) is registered with the model as a separate output tool in order to reduce the complexity of the tool schemas and maximise the chances a model will respond correctly.

If the output type schema is not of type `"object"` (e.g. it's `int` or `list[int]`), the output type is wrapped in a single element object, so the schema of all tools registered with the model are object schemas.

Structured outputs (like tools) use Pydantic to build the JSON schema used for the tool, and to validate the data returned by the model.

!!! note "Type checking considerations"
    The Agent class is generic in its output type, and this type is carried through to `AgentRunResult.output` and `StreamedRunResult.output` so that your IDE or static type checker can warn you when your code doesn't properly take into account all the possible values those outputs could have.

    Static type checkers like pyright and mypy will do their best the infer the agent's output type from the `output_type` you've specified, but they're not always able to do so correctly when you provide functions or multiple types in a union or list, even though PydanticAI will behave correctly. When this happens, your type checker will complain even when you're confident you've passed a valid `output_type`, and you'll need to help the type checker by explicitly specifying the generic parameters on the `Agent` constructor. This is shown in the second example below and the output functions example further down.

    Specifically, there are three valid uses of `output_type` where you'll need to do this:
    1. When using a union of types, e.g. `output_type=Foo | Bar` or in older Python, `output_type=Union[Foo, Bar]`. Until [PEP-747](https://peps.python.org/pep-0747/) "Annotating Type Forms" lands in Python 3.15, type checkers do not consider these a valid value for `output_type`. In addition to the generic parameters on the `Agent` constructor, you'll need to add `# type: ignore` to the line that passes the union to `output_type`.
    2. With mypy: When using a list, as a functionally equivalent alternative to a union, or because you're passing in [output functions](#output-functions). Pyright does handle this correctly, and we've filed [an issue](https://github.com/python/mypy/issues/19142) with mypy to try and get this fixed.
    3. With mypy: when using an async output function. Pyright does handle this correctly, and we've filed [an issue](https://github.com/python/mypy/issues/19143) with mypy to try and get this fixed.

Here's an example of returning either text or structured data:

```python {title="box_or_error.py"}

from pydantic import BaseModel

from pydantic_ai import Agent


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


agent = Agent(
    'openai:gpt-4o-mini',
    output_type=[Box, str],
    system_prompt=(
        "Extract me the dimensions of a box, "
        "if you can't extract all data, ask the user to try again."
    ),
)

result = agent.run_sync('The box is 10x20x30')
print(result.output)
#> Please provide the units for the dimensions (e.g., cm, in, m).

result = agent.run_sync('The box is 10x20x30 cm')
print(result.output)
#> width=10 height=20 depth=30 units='cm'
```

_(This example is complete, it can be run "as is")_

Here's an example of using a union return type, for which PydanticAI will register multiple tools and wraps non-object schemas in an object:

```python {title="colors_or_sizes.py"}
from typing import Union

from pydantic_ai import Agent

agent = Agent[None, Union[list[str], list[int]]](
    'openai:gpt-4o-mini',
    output_type=Union[list[str], list[int]],  # type: ignore
    system_prompt='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.output)
#> ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.output)
#> [10, 20, 30]
```

_(This example is complete, it can be run "as is")_

### Output functions

Instead of plain text or structured data, you may want the output of your agent run to be the result of a function called with arguments provided by the model, for example to further process or validate the data provided through the arguments (with the option to tell the model to try again), or to hand off to another agent.

Output functions are similar to [function tools](tools.md), but the model is forced to call one of them, the call ends the agent run, and the result is not passed back to the model.

As with tool functions, output function arguments provided by the model are validated using Pydantic, they can optionally take [`RunContext`][pydantic_ai.tools.RunContext] as the first argument, and they can raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask the model to try again with modified arguments (or with a different output type).

To specify output functions, you set the agent's `output_type` to either a single function (or bound instance method), or a list of functions. The list can also contain other output types like simple scalars or entire Pydantic models.
You typically do not want to also register your output function as a tool (using the `@agent.tool` decorator or `tools` argument), as this could confuse the model about which it should be calling.

Here's an example of all of these features in action:

```python {title="output_functions.py"}
import re
from typing import Union

from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai._output import ToolRetryError
from pydantic_ai.exceptions import UnexpectedModelBehavior


class Row(BaseModel):
    name: str
    country: str


tables = {
    'capital_cities': [
        Row(name='Amsterdam', country='Netherlands'),
        Row(name='Mexico City', country='Mexico'),
    ]
}


class SQLFailure(BaseModel):
    """An unrecoverable failure. Only use this when you can't change the query to make it work."""

    explanation: str


def run_sql_query(query: str) -> list[Row]:
    """Run a SQL query on the database."""

    select_table = re.match(r'SELECT (.+) FROM (\w+)', query)
    if select_table:
        column_names = select_table.group(1)
        if column_names != '*':
            raise ModelRetry("Only 'SELECT *' is supported, you'll have to do column filtering manually.")

        table_name = select_table.group(2)
        if table_name not in tables:
            raise ModelRetry(
                f"Unknown table '{table_name}' in query '{query}'. Available tables: {', '.join(tables.keys())}."
            )

        return tables[table_name]

    raise ModelRetry(f"Unsupported query: '{query}'.")


sql_agent = Agent[None, Union[list[Row], SQLFailure]](
    'openai:gpt-4o',
    output_type=[run_sql_query, SQLFailure],
    instructions='You are a SQL agent that can run SQL queries on a database.',
)


async def hand_off_to_sql_agent(ctx: RunContext, query: str) -> list[Row]:
    """I take natural language queries, turn them into SQL, and run them on a database."""

    # Drop the final message with the output tool call, as it shouldn't be passed on to the SQL agent
    messages = ctx.messages[:-1]
    try:
        result = await sql_agent.run(query, message_history=messages)
        output = result.output
        if isinstance(output, SQLFailure):
            raise ModelRetry(f'SQL agent failed: {output.explanation}')
        return output
    except UnexpectedModelBehavior as e:
        # Bubble up potentially retryable errors to the router agent
        if (cause := e.__cause__) and isinstance(cause, ToolRetryError):
            raise ModelRetry(f'SQL agent failed: {cause.tool_retry.content}') from e
        else:
            raise


class RouterFailure(BaseModel):
    """Use me when no appropriate agent is found or the used agent failed."""

    explanation: str


router_agent = Agent[None, Union[list[Row], RouterFailure]](
    'openai:gpt-4o',
    output_type=[hand_off_to_sql_agent, RouterFailure],
    instructions='You are a router to other agents. Never try to solve a problem yourself, just pass it on.',
)

result = router_agent.run_sync('Select the names and countries of all capitals')
print(result.output)
"""
[
    Row(name='Amsterdam', country='Netherlands'),
    Row(name='Mexico City', country='Mexico'),
]
"""

result = router_agent.run_sync('Select all pets')
print(result.output)
"""
explanation = "The requested table 'pets' does not exist in the database. The only available table is 'capital_cities', which does not contain data about pets."
"""

result = router_agent.run_sync('How do I fly from Amsterdam to Mexico City?')
print(result.output)
"""
explanation = 'I am not equipped to provide travel information, such as flights from Amsterdam to Mexico City.'
"""
```

### Output validators {#output-validator-functions}

Some validation is inconvenient or impossible to do in Pydantic validators, in particular when the validation requires IO and is asynchronous. PydanticAI provides a way to add validation functions via the [`agent.output_validator`][pydantic_ai.Agent.output_validator] decorator.

If you want to implement separate validation logic for different output types, it's recommended to use [output functions](#output-functions) instead, to save you from having to do `isinstance` checks inside the output validator.

Here's a simplified variant of the [SQL Generation example](examples/sql-gen.md):

```python {title="sql_gen.py"}
from typing import Union

from fake_database import DatabaseConn, QueryError
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry


class Success(BaseModel):
    sql_query: str


class InvalidRequest(BaseModel):
    error_message: str


Output = Union[Success, InvalidRequest]
agent = Agent[DatabaseConn, Output](
    'google-gla:gemini-1.5-flash',
    output_type=Output,  # type: ignore
    deps_type=DatabaseConn,
    system_prompt='Generate PostgreSQL flavored SQL queries based on user input.',
)


@agent.output_validator
async def validate_sql(ctx: RunContext[DatabaseConn], output: Output) -> Output:
    if isinstance(output, InvalidRequest):
        return output
    try:
        await ctx.deps.execute(f'EXPLAIN {output.sql_query}')
    except QueryError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return output


result = agent.run_sync(
    'get me users who were last active yesterday.', deps=DatabaseConn()
)
print(result.output)
#> sql_query='SELECT * FROM users WHERE last_active::date = today() - interval 1 day'
```

_(This example is complete, it can be run "as is")_

## Streamed Results

There two main challenges with streamed results:

1. Validating structured responses before they're complete, this is achieved by "partial validation" which was recently added to Pydantic in [pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748).
2. When receiving a response, we don't know if it's the final response without starting to stream it and peeking at the content. PydanticAI streams just enough of the response to sniff out if it's a tool call or an output, then streams the whole thing and calls tools, or returns the stream as a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult].

### Streaming Text

Example of streamed text output:

```python {title="streamed_hello_world.py" line_length="120"}
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')  # (1)!


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:  # (2)!
        async for message in result.stream_text():  # (3)!
            print(message)
            #> The first known
            #> The first known use of "hello,
            #> The first known use of "hello, world" was in
            #> The first known use of "hello, world" was in a 1974 textbook
            #> The first known use of "hello, world" was in a 1974 textbook about the C
            #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.
```

1. Streaming works with the standard [`Agent`][pydantic_ai.Agent] class, and doesn't require any special setup, just a model that supports streaming (currently all models support streaming).
2. The [`Agent.run_stream()`][pydantic_ai.Agent.run_stream] method is used to start a streamed run, this method returns a context manager so the connection can be closed when the stream completes.
3. Each item yield by [`StreamedRunResult.stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text] is the complete text response, extended as new data is received.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

We can also stream text as deltas rather than the entire text in each item:

```python {title="streamed_delta_hello_world.py"}
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-1.5-flash')


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text(delta=True):  # (1)!
            print(message)
            #> The first known
            #> use of "hello,
            #> world" was in
            #> a 1974 textbook
            #> about the C
            #> programming language.
```

1. [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text] will error if the response is not text.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

!!! warning "Output message not included in `messages`"
    The final output message will **NOT** be added to result messages if you use `.stream_text(delta=True)`,
    see [Messages and chat history](message-history.md) for more information.

### Streaming Structured Output

Not all types are supported with partial validation in Pydantic, see [pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748), generally for model-like structures it's currently best to use `TypeDict`.

Here's an example of streaming a use profile as it's built:

```python {title="streamed_user_profile.py" line_length="120"}
from datetime import date

from typing_extensions import TypedDict

from pydantic_ai import Agent


class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str


agent = Agent(
    'openai:gpt-4o',
    output_type=UserProfile,
    system_prompt='Extract a user profile from the input',
)


async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for profile in result.stream():
            print(profile)
            #> {'name': 'Ben'}
            #> {'name': 'Ben'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

If you want fine-grained control of validation, particularly catching validation errors, you can use the following pattern:

```python {title="streamed_user_profile.py" line_length="120"}
from datetime import date

from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_ai import Agent


class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str


agent = Agent('openai:gpt-4o', output_type=UserProfile)


async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for message, last in result.stream_structured(debounce_by=0.01):  # (1)!
            try:
                profile = await result.validate_structured_output(  # (2)!
                    message,
                    allow_partial=not last,
                )
            except ValidationError:
                continue
            print(profile)
            #> {'name': 'Ben'}
            #> {'name': 'Ben'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
```

1. [`stream_structured`][pydantic_ai.result.StreamedRunResult.stream_structured] streams the data as [`ModelResponse`][pydantic_ai.messages.ModelResponse] objects, thus iteration can't fail with a `ValidationError`.
2. [`validate_structured_output`][pydantic_ai.result.StreamedRunResult.validate_structured_output] validates the data, `allow_partial=True` enables pydantic's [`experimental_allow_partial` flag on `TypeAdapter`][pydantic.type_adapter.TypeAdapter.validate_json].

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Examples

The following examples demonstrate how to use streamed responses in PydanticAI:

- [Stream markdown](examples/stream-markdown.md)
- [Stream Whales](examples/stream-whales.md)
