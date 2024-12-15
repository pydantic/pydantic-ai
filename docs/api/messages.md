# `pydantic_ai.messages`

The structure of [Message][pydantic_ai.messages.Message] can be shown as a graph:

```mermaid
graph RL
    SystemPrompt(SystemPrompt) --- ModelRequestPart
    UserPrompt(UserPrompt) --- ModelRequestPart
    ToolReturn(ToolReturn) --- ModelRequestPart
    RetryPrompt(RetryPrompt) --- ModelRequestPart
    TextPart(TextPart) --- ModelResponsePart
    ToolCallPart(ToolCallPart) --- ModelResponsePart
    ModelRequestPart("ModelRequestPart<br>(Union)") --- ModelRequest
    ModelRequest("ModelRequest(parts=list[...])") --- ModelMessage
    ModelResponsePart("ModelResponsePart<br>(Union)") --- ModelResponse
    ModelResponse("ModelResponse(parts=list[...])") --- ModelMessage("ModelMessage<br>(Union)")
```

::: pydantic_ai.messages
