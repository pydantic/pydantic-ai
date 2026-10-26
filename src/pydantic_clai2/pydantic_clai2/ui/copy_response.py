"""`/copy`: the Markdown source of a model response, put on the clipboard."""

from collections.abc import Callable, Sequence

from pydantic_ai.messages import ModelMessage, ModelResponse

USAGE = 'Usage: /copy [N]'


def response_texts(messages: Sequence[ModelMessage]) -> list[str]:
    """The text of each response that has any, latest first, without thinking or tool calls."""
    return [
        text
        for message in reversed(messages)
        if isinstance(message, ModelResponse) and (text := message.text) is not None and text.strip()
    ]


def copy_command(args: list[str], *, messages: Sequence[ModelMessage], copy: Callable[[str], None]) -> str:
    """`/copy [N]`: copy the `N`th-latest response's text, by default the last one, and say what was copied."""
    if len(args) > 1 or (args and not (args[0].isdecimal() and int(args[0]) > 0)):
        raise ValueError(USAGE)
    nth = int(args[0]) if args else 1
    texts = response_texts(messages)
    if not texts:
        return 'No response to copy yet.'
    if nth > len(texts):
        return f'There {"is" if len(texts) == 1 else "are"} only {_count(len(texts), "response")} to copy.'
    text = texts[nth - 1]
    copy(text)
    which = 'the last response' if nth == 1 else f'response {nth} from the end'
    return f'Copied {which} ({_count(len(text), "character")}).'


def _count(number: int, noun: str) -> str:
    return f'{number:,} {noun}{"" if number == 1 else "s"}'
