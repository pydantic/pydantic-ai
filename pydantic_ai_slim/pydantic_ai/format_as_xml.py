from typing_extensions import deprecated

from .format_prompt import format_as_xml as _format_as_xml


@deprecated('`format_as_xml` has moved, import it via `from pydantic_ai import format_as_xml`')
def format_as_xml(prompt: str) -> str:
    """`format_as_xml` has moved, import it via `from pydantic_ai import format_as_xml` instead."""
    return _format_as_xml(prompt)
