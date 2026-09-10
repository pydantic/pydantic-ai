"""An agent spec fails at load time, not at runtime.

Templates are validated against typed deps when the spec is built: a single
typo error names the field and the file. The same spec typechecks into a
running agent offline.
"""
from pydantic import BaseModel
from pydantic_ai import Agent, AgentSpec


class UserContext(BaseModel):
    user_name: str
    user_role: str


bad = {
    'name': 'support',
    'model': 'test',  # offline stub backend
    'instructions': 'You are {{non_existent_field}}. Be nice.',
    'tools': [],
    'capabilities': [],
}
def main() -> None:
    try:
        Agent.from_spec(bad, deps_type=UserContext)
        print('BUG: invalid template accepted')
    except Exception as exc:
        print(f'{type(exc).__name__}: {str(exc)[:100]}')


    good = {
        'name': 'support',
        'model': 'test',
        'instructions': 'You are {{user_role}} {{user_name}}. Be nice.',
        'tools': [],
        'capabilities': [],
    }
    agent = Agent.from_spec(good, deps_type=UserContext)
    print(f'valid template -> {type(agent).__name__}({agent.name!r}) runs offline')


if __name__ == '__main__':
    main()
