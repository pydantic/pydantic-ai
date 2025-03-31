import asyncio
from datetime import datetime

import logfire
from logfire import ConsoleOptions

from pydantic_ai_examples.evals import infer_time_range
from pydantic_ai_examples.evals.models import TimeRangeInputs

if __name__ == '__main__':

    async def main():
        """Example usage of the time range inference agent."""
        logfire.configure(
            send_to_logfire='if-token-present', console=ConsoleOptions(verbose=True)
        )
        user_prompt = 'yesterday from 2-4 ET'
        # user_prompt = 'the last 24 hours'
        # user_prompt = '6 to 9 PM ET on October 8th'
        # user_prompt = 'next week'
        # user_prompt = 'what time is it?'

        print(
            await infer_time_range(
                TimeRangeInputs(prompt=user_prompt, now=datetime.now().astimezone())
            )
        )

    asyncio.run(main())
