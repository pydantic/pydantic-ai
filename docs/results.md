Results are the final values returned from [running an agent](agents.md#running-agents).
The result values are wrapped in [`RunResult`][pydantic_ai.result.RunResult] and [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] so you can access other data like [cost][pydantic_ai.result.Cost] of the run and [message history](message-history.md#accessing-messages-from-results)

Both `RunResult` and `StreamedRunResult` are generic in the data they wrap, so typing information about the data returned by the agent is preserved.

```py title="olympics.py"
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('gemini-1.5-flash', result_type=CityLocation)
result = agent.run_sync('Where the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.cost())
#> Cost(request_tokens=56, response_tokens=8, total_tokens=64, details=None)
```

Runs end when either a plain text response is received or the model calls a tool associated with one of the structured result types. We will add limits to make sure a run doesn't go on indefinitely, see [#70](https://github.com/pydantic/pydantic-ai/issues/70).

## Result data {#structured-result-validation}

When the result type is `str`, or a union including `str`, plain text responses are enabled on the model, and the raw text response from the model is used as the response data.

If the result type is a union with multiple members (after remove `str` from the members), each member is registered as a separate tool with the model in order to reduce the complexity of the tool schemas and maximise the changes a model will respond correctly.

If the result type schema is not of type `"object"`, the result type is wrapped in a single element object, so the schema of all tools registered with the model are object schemas.

Structured results (like retrievers) use Pydantic to build the JSON schema used for the tool, and to validate the data returned by the model.

!!! note "Bring on PEP-747"
    Until [PEP-747](https://peps.python.org/pep-0747/) "Annotating Type Forms" lands, unions are not valid as `type`s in Python.

    When creating the agent we nee to `# type: ignore` the `result_type` argument, and add a type hint to tell type checkers about the type of the agent.

Here's an example of returning either text or a structured value

```py title="box_or_error.py"
from typing import Union

from pydantic import BaseModel

from pydantic_ai import Agent


class Box(BaseModel):
    width: int
    height: int
    depth: int
    units: str


agent: Agent[None, Union[Box, str]] = Agent(
    'openai:gpt-4o-mini',
    result_type=Union[Box, str],  # type: ignore
    system_prompt=(
        "Extract me the dimensions of a box, "
        "if you can't extract all data, ask the user to try again."
    ),
)

result = agent.run_sync('The box is 10x20x30')
print(result.data)
#> Please provide the units for the dimensions (e.g., cm, in, m).

result = agent.run_sync('The box is 10x20x30 cm')
print(result.data)
#> width=10 height=20 depth=30 units='cm'
```

Here's an example of using a union return type which registered multiple tools, and wraps non-object schemas in an object:

```py title="colors_or_sizes.py"
from typing import Union

from pydantic_ai import Agent

agent: Agent[None, Union[list[str], list[int]]] = Agent(
    'openai:gpt-4o-mini',
    result_type=Union[list[str], list[int]],  # type: ignore
    system_prompt='Extract either colors or sizes from the shapes provided.',
)

result = agent.run_sync('red square, blue circle, green triangle')
print(result.data)
#> ['red', 'blue', 'green']

result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.data)
#> [10, 20, 30]
```

### Result validators functions

Some validation is inconvenient or impossible to do in Pydantic validators, in particular when the validation requires IO and is asynchronous. PydanticAI provides a way to add validation functions via the [`agent.result_validator`][pydantic_ai.Agent.result_validator] decorator.

Here's a simplified variant of the [SQL Generation example](examples/sql-gen.md):

```py title="sql_gen.py"
from typing import Union

from fake_database import DatabaseConn, QueryError
from pydantic import BaseModel

from pydantic_ai import Agent, CallContext, ModelRetry


class Success(BaseModel):
    sql_query: str


class InvalidRequest(BaseModel):
    error_message: str


Response = Union[Success, InvalidRequest]
agent: Agent[DatabaseConn, Response] = Agent(
    'gemini-1.5-flash',
    result_type=Response,  # type: ignore
    deps_type=DatabaseConn,
    system_prompt='Generate PostgreSQL flavored SQL queries based on user input.',
)


@agent.result_validator
async def validate_result(ctx: CallContext[DatabaseConn], result: Response) -> Response:
    if isinstance(result, InvalidRequest):
        return result
    try:
        await ctx.deps.execute(f'EXPLAIN {result.sql_query}')
    except QueryError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return result


result = agent.run_sync(
    'get me uses who were last active yesterday.', deps=DatabaseConn()
)
print(result.data)
#> sql_query='SELECT * FROM users WHERE last_active::date = today() - interval 1 day'
```

## Streamed Results

**TODO**

Streamed responses provide a unique challenge:
* validating the partial result is both practically and semantically complex, but pydantic can do this
* we don't know if a result will be the final result of a run until we start streaming it, so PydanticAI has to start streaming just enough of the response to sniff out if it's the final response, then either stream the rest of the response to call a retriever, or return an object that lets the rest of the response be streamed by the user
* examples including: streaming text, streaming validated data, streaming the raw data to do validation inside a try/except block when necessary
* explanation of how streamed responses are "debounced"
