"""A tool may stop the run. ctx.cancel() requests cancellation; the run ends
in a catchable RunCancelled carrying everything completed before it stopped.
"""
import asyncio
from pydantic_ai import Agent, RunCancelled
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart


async def model(messages, info):
    return ModelResponse(parts=[ToolCallPart('slow_job', {})])


agent = Agent(FunctionModel(model))


@agent.tool
async def slow_job(ctx) -> str:
    ctx.cancel()  # cooperative: returns normally, lands at the next await
    await asyncio.sleep(0)
    return 'never used'


def main():
    try:
        agent.run_sync('start the job')
        print('BUG: run completed')
    except RunCancelled as exc:
        history = exc.all_messages()
        print(f'run ended with RunCancelled; completed work preserved ({len(history)} message(s))')
        print('cancellation is a typed, catchable, resumable outcome')
        assert len(history) >= 1
if __name__ == '__main__':
    main()
