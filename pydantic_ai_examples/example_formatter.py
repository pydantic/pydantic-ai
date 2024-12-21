import os
from dataclasses import dataclass
from typing import cast

import logfire
from devtools import debug
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.format_examples import format_examples
from pydantic_ai.models import KnownModelName

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')


class TextSentiment(BaseModel):
    sentiment: str


model = cast(KnownModelName, os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'))
print(f'Using model: {model}')


@dataclass
class SupportDependencies:
    text: str


agent = Agent(model, deps_type=SupportDependencies, result_type=TextSentiment)

# Example Usage:
if __name__ == '__main__':
    dict_example = {
        'text': "I absolutely love this car! It's smooth and efficient.",
        'sentiment': 'Positive',
    }

    class pydantic_example(BaseModel):
        text: str
        sentiment: str

    @dataclass
    class dataclass_example:
        text: str
        sentiment: str

    examples = [
        dict_example,  # dict
        pydantic_example(
            text='The design of this model is terrible, and the fuel efficiency is poor.',
            sentiment='Negative',
        ),  # pydantic model
        dataclass_example(
            text='The car is okay, nothing special. Just an average experience.',
            sentiment='Neutral',
        ),  # dataclass
    ]

    @agent.system_prompt
    def get_sentiment(ctx: RunContext[SupportDependencies]) -> str:
        return f'Take these examples to asses sentiment of the text:\n{format_examples(examples, dialect='xml')}'

    user_text: str = 'The interior is spacious and feels luxurious, but the mileage is disappointing.'

    result = agent.run_sync(
        'what is the sentiment?', deps=SupportDependencies(text=user_text)
    )
    debug(result)
    print(result.data)
