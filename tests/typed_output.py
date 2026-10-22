"""Pins that pyright still refuses an `output_type` that isn't a type form.

Unlike `typed_agent.py`, this module keeps `reportUnnecessaryTypeIgnoreComment` on, so an ignore below
that stops being needed fails the type check.
"""

from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class Foo:
    a: int


def refused_output_types() -> None:
    # Never called: these raise at run time, and only pyright's verdict is pinned here.
    Agent(output_type=5)  # pyright: ignore[reportArgumentType,reportCallIssue]
    Agent(output_type=Foo(a=1))  # pyright: ignore[reportArgumentType,reportCallIssue]
    Agent(output_type=[Foo, 0])  # pyright: ignore[reportArgumentType,reportCallIssue]
