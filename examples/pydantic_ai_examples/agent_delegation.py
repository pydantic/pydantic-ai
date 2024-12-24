"""Example of multi-agent flow where one agent delegates work to another.

In this scenario, a group of agents work together to find flights for a user.
"""

import datetime
from dataclasses import dataclass

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.result import Usage
from pydantic_ai.settings import UsageLimits

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str | None = None
    price: int
    origin: str = Field(description='Three-letter airport code')
    destination: str = Field(description='Three-letter airport code')
    date: datetime.date


class NoFlightFound(BaseModel):
    """When no valid flight is found."""


@dataclass
class Deps:
    web_page_text: str
    req_origin: str
    req_destination: str
    req_date: datetime.date


# This agent is responsible for controlling the flow of the conversation.
flights_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'openai:gpt-4o',
    deps_type=Deps,
    result_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        'Your job is to find the cheapest flight for the user on the given date. '
    ),
)


# This agent is responsible for extracting flight details from text.
search_agent = Agent(
    'openai:gpt-4o',
    result_type=list[FlightDetails],
    system_prompt=('Extract all the flight details from the given text.'),
)


@flights_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all flights."""
    # we pass the usage to the search agent so requests within this agent are counted
    result = await search_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
    return result.data


@flights_agent.result_validator
async def validate_result(
    ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(result, NoFlightFound):
        return result

    errors: list[str] = []
    if result.origin != ctx.deps.req_origin:
        errors.append(
            f'Flight should have origin {ctx.deps.req_origin}, not {result.origin}'
        )
    if result.destination != ctx.deps.req_destination:
        errors.append(
            f'Flight should have destination {ctx.deps.req_destination}, not {result.destination}'
        )
    if result.date != ctx.deps.req_date:
        errors.append(f'Flight should be on {ctx.deps.req_date}, not {result.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return result


flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""


async def main():
    deps = Deps(
        web_page_text=flights_web_page,
        req_origin='SFO',
        req_destination='ANC',
        req_date=datetime.date(2025, 1, 10),
    )
    message_history: list[ModelMessage] | None = None
    usage: Usage = Usage()
    while True:
        result = await flights_agent.run(
            'Find me a flight',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=UsageLimits(request_limit=15),
        )
        if isinstance(result.data, NoFlightFound):
            print('No flight found')
            break
        else:
            print(f'Flight found: {result.data}')
            answer = input(
                'Do you want to buy this flight, or keep searching? (buy/no): '
            )
            if answer == 'buy':
                print('Purchasing flight...')
                break
            result.set_result_tool_return('Please suggest another flight')
            message_history = result.all_messages()


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
