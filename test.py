from pydantic_ai import Agent, RunContext
from typing_extensions import TypedDict
from typing import Literal

class GameInformation(TypedDict):
    balance: int
    winning_square: int

class GameResult(TypedDict):
    balance: int
    result: Literal['win', 'loss']


roulette_agent = Agent(
    'claude-3-5-haiku-latest',
    deps_type=GameInformation,
    result_type=GameResult,
    system_prompt=(
        'You are a roulette dealer. Use the `validate_bet` tool to ensure the bet is valid. '
        'If the bet is valid (`validate_bet` returns `True`), use the `roulette_wheel` tool to determine if the player has won. '
        'If the bet is invalid (`validate_bet` returns `False`), you must return an error message, and do not call the `roulette_wheel` function.'
        'Players start with a balance and place a bet amount on a specific square. '
        'If they win, they earn 35x their bet; if they lose, they lose their bet amount. '
        'Return the updated balance and the win/loss status.'
    ),
)


@roulette_agent.tool
async def validate_bet(ctx: RunContext[GameInformation], square: int, bet: int) -> bool:
    """Validate that the bet is within the player's balance and the square is valid."""
    balance = ctx.deps["balance"]
    if bet > balance:
        return False
    if square < 0 or square > 100:
        return False
    return True


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[GameInformation], square: int, bet: int) -> GameResult:
    """Spin the roulette wheel and update the player's balance based on the result."""
    winning_square = ctx.deps["winning_square"]
    balance = ctx.deps["balance"]

    if square == winning_square:
        winnings = bet * 35
        balance += winnings
        return {"balance": balance, "result": "win"}
    else:
        balance -= bet
        return {"balance": balance, "result": "loss"}


deps: GameInformation = {"balance": 100, "winning_square": 18}

# Valid bet
# result = roulette_agent.run_sync('Bet 10 on square eightneen', deps=deps)
# print(result.data)
# #> {'balance': 450, 'result': 'win'}

# Invalid bet
result = roulette_agent.run_sync('Bet 200 on square five',  deps=deps)
print(result.data)
#> {'balance': -100, 'result': 'loss'}

# # Losing bet
# result = roulette_agent.run_sync('Bet twenty on square five', deps=deps)
# print(result.data)
# #> {'balance': 80, 'result': 'loss'}

print(roulette_agent.last_run_messages)
