"""v1: Agent(builtin_tools=...) ctor kwarg is deprecated."""
from pydantic_ai import Agent


def trigger():
    # DEPRECATION: G_builtin_tools
    Agent('test', builtin_tools=[])


EXPECT = '`Agent(builtin_tools=...)` is deprecated'

if __name__ == '__main__':
    trigger()
