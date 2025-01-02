from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta

import logfire
from devtools import debug
from pydantic import BaseModel
from pydantic_ai_graph import AbstractState, BaseNode, End, Graph, GraphContext

from pydantic_ai import Agent, RunContext

logfire.configure(send_to_logfire='if-token-present')


class EventDetails(BaseModel):
    title: str
    location: str
    start_ts: datetime
    end_ts: datetime


class State(AbstractState, BaseModel):
    email_content: str
    skip_events: list[str] = []
    attempt: int = 0

    def serialize(self) -> bytes | None:
        return self.model_dump_json(exclude={'email_content'}).encode()


class RawEventDetails(BaseModel):
    title: str
    location: str
    start_ts: str
    duration: str


extract_agent = Agent('openai:gpt-4o', result_type=RawEventDetails, deps_type=list[str])


@extract_agent.system_prompt
def extract_system_prompt(ctx: RunContext[list[str]]):
    prompt = 'Extract event details from the email body.'
    if ctx.deps:
        skip_events = '\n'.join(ctx.deps)
        prompt += f'\n\nDo not return the following events:\n{skip_events}'
    return prompt


@dataclass
class ExtractEvent(BaseNode[State]):
    async def run(self, ctx: GraphContext[State]) -> CleanEvent:
        event = await extract_agent.run(
            ctx.state.email_content, deps=ctx.state.skip_events
        )
        return CleanEvent(event.data)


# agent used to extract the timestamp from the string in `CleanEvent`
timestamp_agent = Agent('openai:gpt-4o', result_type=datetime)


@timestamp_agent.system_prompt
def timestamp_system_prompt():
    return f'Extract the timestamp from the string, the current timestamp is: {datetime.now().isoformat()}'


# agent used to extract the duration from the string in `CleanEvent`
duration_agent = Agent(
    'openai:gpt-4o',
    result_type=timedelta,
    system_prompt='Extract the duration from the string as an ISO 8601 interval.',
)


@dataclass
class CleanEvent(BaseNode[State]):
    input_data: RawEventDetails

    async def run(self, ctx: GraphContext[State]) -> InspectEvent:
        start_ts, duration = await asyncio.gather(
            timestamp_agent.run(self.input_data.start_ts),
            duration_agent.run(self.input_data.duration),
        )
        return InspectEvent(
            EventDetails(
                title=self.input_data.title,
                location=self.input_data.location,
                start_ts=start_ts.data,
                end_ts=start_ts.data + duration.data,
            )
        )


@dataclass
class InspectEvent(BaseNode[State, EventDetails | None]):
    input_data: EventDetails

    async def run(
        self, ctx: GraphContext[State]
    ) -> ExtractEvent | End[EventDetails | None]:
        now = datetime.now()
        if self.input_data.start_ts.tzinfo is not None:
            now = now.astimezone(self.input_data.start_ts.tzinfo)

        if self.input_data.start_ts > now:
            return End(self.input_data)
        ctx.state.attempt += 1
        if ctx.state.attempt > 2:
            return End(None)
        else:
            ctx.state.skip_events.append(self.input_data.title)
            return ExtractEvent()


graph = Graph[State, EventDetails | None](
    nodes=(
        ExtractEvent,
        CleanEvent,
        InspectEvent,
    )
)
graph_runner = graph.get_runner(ExtractEvent)
print(graph_runner.mermaid_code())

email = """
Hi Samuel,

I hope this message finds you well! I wanted to share a quick update on our recent and upcoming team events.

Firstly, a big thank you to everyone who participated in last month's
Team Building Retreat held on November 15th 2024 for 1 day.
It was a fantastic opportunity to enhance our collaboration and communication skills while having fun. Your
feedback was incredibly positive, and we're already planning to make the next retreat even better!

Looking ahead, I'm excited to invite you all to our Annual Year-End Gala on January 20th 2025.
This event will be held at the Grand City Ballroom starting at 6 PM until 8pm. It promises to be an evening full
of entertainment, good food, and great company, celebrating the achievements and hard work of our amazing team
over the past year.

Please mark your calendars and RSVP by January 10th. I hope to see all of you there!

Best regards,
"""


async def main():
    state = State(email_content=email)
    result, history = await graph_runner.run(state, None)
    debug(result, history)


if __name__ == '__main__':
    asyncio.run(main())
