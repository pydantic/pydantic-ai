"""The deps boundary: the model never sees trusted state.

The password lives in deps. Only the tool may read it. The model only ever
receives tool definitions; its request payload contains no secret.
"""
import asyncio
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

DB = {'db_password': 'hunter2-keep-secret'}


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('check_db', {'key': 'readiness probe'})])
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(model), deps_type=dict)


@agent.tool
async def check_db(ctx, key: str) -> str:
    ok = ctx.deps['db_password'] == 'hunter2-keep-secret'
    return f'db:{key}:{"ok" if ok else "auth-failed"}'  # never echoes the secret


async def main():
    with capture_run_messages() as msgs:
        result = await agent.run('Is the db ready?', deps=DB)
    payload = str(msgs)
    leaked = 'hunter2' in payload
    assert not leaked, 'the secret crossed into model-visible messages!'
    print(f'request payload contained the db password: {leaked}')
    print(f'tool executed with deps ({result.output!r}); model saw only tool definitions')


if __name__ == '__main__':
    asyncio.run(main())
