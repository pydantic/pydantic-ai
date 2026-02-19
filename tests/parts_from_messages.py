from typing import Any

from pydantic_ai import ModelMessage, ModelRequestPart, ModelResponsePart


def part_types_from_messages(messages: list[ModelMessage]) -> list[Any]:
    """Utility function used when you are not interested in the content of the messages, but only that the part is there.

    As an example, the following messages:

    ```python
    [
        ModelRequest(parts=[UserPromptPart(content='')], kind='request'),
        ModelResponse(parts=[TextPart(content='')], kind='response'),
    ]
    ```

    Will return:

    ```python
    [
        [UserPromptPart],
        [TextPart],
    ]
    ```

    So each list represents either `ModelRequest` or `ModelResponse` and the parts that are present in the message.
    """
    parts: list[Any] = []
    for message in messages:
        message_parts: list[type[ModelResponsePart | ModelRequestPart]] = []
        for part in message.parts:
            message_parts.append(type(part))
        parts.append(message_parts)
    return parts
