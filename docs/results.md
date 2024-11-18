## Ending runs

**TODO**

* runs end when either a plain text response is received or the model calls a tool associated with one of the structured result types
* example
* we should add `message_limit` (number of model messages) and `cost_limit` to `run()` etc.

## Structured result validation

**TODO**

* structured results (like retrievers) use Pydantic, Pydantic builds the JSON schema and does the validation
* PydanticAI tries hard to simplify the schema, this means:
  * if the return type is `str` or a union including `str`, plain text responses are enabled
  * if the schema is a union (after remove `str` from the members), each member is registered as its own tool call
  * if the schema is not an object, the result type is wrapped in a single element object

## Result validators functions

**TODO**

* Some validation is inconvenient or impossible to do in Pydantic validators, in particular when the validation requires IO and is asynchronous. PydanticAI provides a way to add validation functions via the [`agent.result_validator`][pydantic_ai.Agent.result_validator] decorator.
* example

## Streamed Results

**TODO**

Streamed responses provide a unique challenge:
* validating the partial result is both practically and semantically complex, but pydantic can do this
* we don't know if a result will be the final result of a run until we start streaming it, so PydanticAI has to start streaming just enough of the response to sniff out if it's the final response, then either stream the rest of the response to call a retriever, or return an object that lets the rest of teh response be streamed by the user
* examples including: streaming text, streaming validated data, streaming the raw data to do validation inside a try/except block when necessary
* explanation of how streamed responses are "debounced"

## Cost

**TODO**

* counts tokens, not dollars
* example
