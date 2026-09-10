"""A usage limit stops a run BEFORE a side-effect batch executes.

The model asks for two tool calls in one response; the limit allows one.
The whole batch is rejected, so neither tool runs: budget checks precede
execution, not polite suggestions after it.
"""
from pydantic_ai import Agent, UsageLimits, UsageLimitExceeded
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart

side_effects = []


async def model(messages, info):
    return ModelResponse(
        parts=[
            ToolCallPart('credit_customer', {'amount': 100}),
            ToolCallPart('credit_customer', {'amount': 100}),
        ]
    )


agent = Agent(FunctionModel(model))


@agent.tool
def credit_customer(ctx, amount: int) -> str:
    side_effects.append(('credited', amount))
    return 'ok'


def main() -> None:
    try:
        agent.run_sync('credit the customer twice', usage_limits=UsageLimits(tool_calls_limit=1))
        print('BUG: exceeded the limit')
    except UsageLimitExceeded as exc:
        print(f'{type(exc).__name__}: {str(exc)[:60]}...')
        print(f'tool executions that happened: {len(side_effects)}')
        assert not side_effects, 'a side effect ran despite the budget'


if __name__ == '__main__':
    main()
