# Plan: OpenAI Conversations API state support

## Issue

Refs https://github.com/pydantic/pydantic-ai/issues/5222

OpenAI's Responses API supports durable conversation state through the `conversation` request parameter and the Conversations API. Pydantic AI already supports server-side state with `openai_previous_response_id`, but it does not expose `conversation` or trim message history when a durable conversation ID is being reused.

Without trimming, a later request that uses the same OpenAI conversation can resend input/output items that OpenAI already has in the conversation object, duplicating stored state.

## Proposed API

Add a Responses-only model setting:

```python
OpenAIResponsesModelSettings(
    openai_conversation_id='conv_...',
)
```

Also support:

```python
OpenAIResponsesModelSettings(
    openai_conversation_id='auto',
)
```

`'auto'` would mean: use the most recent same-provider `ModelResponse` in message history whose `provider_details['conversation_id']` is a string.

## Behavior

### Request construction

When `openai_conversation_id` resolves to a concrete ID, pass it to `client.responses.create()` as:

```python
conversation='conv_...'
```

The OpenAI Python SDK currently accepts either a string conversation ID or `{'id': ...}`; the direct string form matches the public API examples and keeps the request simple.

Do not apply this setting to `responses.compact()`. The standalone compact endpoint is stateless and its SDK params expose `previous_response_id`, but not `conversation`.

### Response metadata

When OpenAI returns a response with `response.conversation.id`, store it on `ModelResponse.provider_details`:

```python
provider_details={
    ...,
    'conversation_id': 'conv_...',
}
```

This should be applied to both non-streaming and streaming Responses paths. The issue text says `provider_metadata`, but Pydantic AI's message model uses `provider_details`, so this should follow the existing provider-specific metadata convention.

### History trimming

For a concrete conversation ID:

1. Search message history from newest to oldest.
2. Find the most recent `ModelResponse` where:
   - `response.provider_name == self.system`
   - `response.provider_details`
   - `response.provider_details.get('conversation_id') == openai_conversation_id`
3. If found, omit all messages before and including that response.
4. Send only the messages that came after it as new input items.
5. If no matching response exists, send the full history with the configured conversation ID.

For `'auto'`:

1. Search message history from newest to oldest.
2. Find the most recent same-provider `ModelResponse` with a string `provider_details['conversation_id']`.
3. If found, use that conversation ID and trim before and including the response.
4. If not found, do not send `conversation`, and send the full history.

This mirrors `openai_previous_response_id='auto'`, but keys the continuation on the durable conversation ID returned by OpenAI rather than a response ID.

### Interaction with `openai_previous_response_id`

OpenAI documents `previous_response_id` and `conversation` as mutually exclusive. I propose Pydantic AI should raise `UserError` when both `openai_previous_response_id` and `openai_conversation_id` are set on the same request, rather than trying to resolve precedence implicitly.

The empty-input fallback in `_responses_create()` should treat a resolved conversation ID like a resolved previous response ID: if there are no new input messages but conversation state is present, do not inject an empty user message just to satisfy stateless request validation.

### Compaction

Stateful compaction through `context_management` should continue to work with the regular `/responses` request because it is just another request parameter.

Stateless `OpenAICompaction(stateless=True)` should remain independent of `openai_conversation_id` because it calls `responses.compact()` and returns a compaction item that is round-tripped through Pydantic AI message history.

## Implementation outline

1. Add `openai_conversation_id: Literal['auto'] | str` to `OpenAIResponsesModelSettings` with a docstring that explains concrete IDs, `'auto'`, metadata storage, and the `previous_response_id` incompatibility.
2. Add a small private resolver on `OpenAIResponsesModel`, likely parallel to `_resolve_previous_response_id`, that returns `(conversation_id, trimmed_messages)`.
3. In `_responses_create()`:
   - fail fast if both server-side state settings are set
   - resolve conversation state before `_map_messages()`
   - pass `conversation=conversation_id or OMIT`
   - keep `previous_response_id` unset whenever conversation is set
   - update the empty-input fallback condition
4. In `_process_response()`, merge returned `response.conversation.id` into `provider_details`.
5. In `OpenAIResponsesStreamedResponse`, capture `chunk.response.conversation.id` from streamed response events and merge it into `provider_details`.
6. Update `docs/models/openai.md` near "Referencing earlier responses" with a durable Conversations API subsection.

## Tests

Add tests in `tests/models/test_openai_responses.py`, using the existing mock OpenAI Responses client where practical and testing through `Agent`/model request APIs rather than private helpers.

Planned coverage:

- request construction passes `conversation='conv_...'`
- returned `response.conversation.id` is stored as `provider_details['conversation_id']`
- same conversation ID trims earlier history and sends only new input items
- different conversation IDs do not trim unrelated history
- different providers do not trim unrelated history, even when `conversation_id` matches
- `openai_conversation_id='auto'` resolves from same-provider response history
- `openai_conversation_id='auto'` with no matching history sends no `conversation`
- setting both `openai_conversation_id` and `openai_previous_response_id` raises `UserError`
- tool-call continuations and `ModelRetry` retries do not resend the original user prompt after a response has been associated with the conversation
- streaming responses also preserve `conversation_id` in final `ModelResponse.provider_details`

## Open questions

1. Should Pydantic AI always fail when both `openai_conversation_id` and `openai_previous_response_id` are present, even if one is `'auto'` and would resolve to no concrete value?
2. Should `openai_conversation_id='auto'` only reuse an existing ID from history, or should Pydantic AI ever create a new OpenAI conversation object? I propose reuse-only; creation can stay with user code via the OpenAI client.
3. Should concrete `openai_conversation_id='conv_...'` trim only when history contains matching provider metadata, or should it also trim against any same-provider response when the response does not yet have conversation metadata? I propose matching metadata only, to avoid dropping context that may not actually be in the OpenAI conversation.
