from __future__ import annotations

from textwrap import dedent
from typing import Annotated, Any

import modal
from annotated_types import Ge, Le
from pydantic import BaseModel

NEW_LEAD_CHANNEL = '#hackathon-sales-qual-agent'
DAILY_SUMMARY_CHANNEL = '#hackathon-sales-qual-agent'

image = modal.Image.debian_slim(python_version='3.13').pip_install(
    'pydantic',
    'pydantic_ai_slim[openai,duckduckgo]',
    'logfire[httpx,fastapi]',
    'fastapi[standard]',
    'httpx',
)
app = modal.App(
    name='slack-lead-qualifier',
    image=image,
    secrets=[
        modal.Secret.from_name('logfire'),
        modal.Secret.from_name('openai'),
        modal.Secret.from_name('slack'),
    ],
)


@app.function(min_containers=1)
@modal.asgi_app()
def web_app():
    return SlackLeadQualifier().web_app


@app.function()
async def process_slack_member(profile: Profile, logfire_ctx: Any):
    from logfire.propagate import attach_context

    with attach_context(logfire_ctx):
        return await SlackLeadQualifier().process_slack_member(profile)


@app.function(schedule=modal.Cron('0 8 * * *'))  # Every day at 8am
async def send_daily_summary():
    await SlackLeadQualifier().send_daily_summary()


class Profile(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    email: str

    def as_prompt(self) -> str:
        from pydantic_ai import format_as_xml

        return format_as_xml(self, root_tag='profile')


class Analysis(BaseModel):
    profile: Profile
    organization_name: str
    organization_domain: str
    job_title: str
    relevance: Annotated[int, Ge(1), Le(5)]
    """Relevance as a sales lead on a scale of 1 to 5"""
    summary: str
    """Short summary of the user and their relevance."""

    def as_slack_blocks(self, include_relevance: bool = False) -> list[dict[str, Any]]:
        profile = self.profile
        relevance = f'({self.relevance}/5)' if include_relevance else ''
        return [
            {
                'type': 'markdown',
                'text': f'[{profile.display_name}](mailto:{profile.email}), {self.job_title} at [**{self.organization_name}**](https://{self.organization_domain}) {relevance}',
            },
            {
                'type': 'markdown',
                'text': self.summary,
            },
        ]


class Unknown(BaseModel):
    reason: str
    """Reason for why you couldn't find anything."""


class AnalysisStore:
    def __init__(self):
        self.store = modal.Dict.from_name('analyses', create_if_missing=True)

    async def add(self, analysis: Analysis):
        await self.store.put.aio(analysis.profile.email, analysis.model_dump())

    async def list(self) -> list[Analysis]:
        return [
            Analysis.model_validate(analysis)
            async for analysis in self.store.values.aio()
        ]

    async def clear(self):
        await self.store.clear.aio()


class SlackLeadQualifier:
    def __init__(self):
        self.logfire = self.setup_logfire()
        self.agent = self.build_agent()
        self.web_app = self.build_web_app()
        self.store = AnalysisStore()

    def setup_logfire(self):
        import logfire as _logfire

        logfire = _logfire.configure(service_name=app.name)
        logfire.instrument_pydantic_ai()
        logfire.instrument_httpx(capture_all=True)
        return logfire

    def build_agent(self):
        from pydantic_ai import Agent, NativeOutput
        from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

        agent = Agent(
            'openai:gpt-4o',
            instructions=dedent(
                """
                Your job is to evaluate a user who's joined our public Slack, and provide a summary of how important they might be as a customer to us.

                Our company, Pydantic, offers three products:
                * Pydantic Validation: A powerful library for data validation in Python (free and open source)
                * Pydantic AI: A Python Agent Framework (free and open source)
                * Pydantic Logfire: a general purpose observability framework (Traces, Logs and Metrics) with special support for
                Python, Javascript/TypeScript and Rust. It's particularly useful in AI applications. (commercial paid product)

                We particularly want to find developers working for or running prominent/well funded companies that might pay for Pydantic Logfire.

                Always use your search tool to research the user and the company they work for, based on the email domain or what you find on e.g. LinkedIn and GitHub.
                Note that our products are aimed at software developers, data scientists, and AI engineers, so if the person you find is not in a technical role,
                you're likely looking at the wrong person. In that case, you should search again with additional keywords to narrow it down to developers.

                If you couldn't find anything useful, return Unknown.
                """
            ),
            tools=[duckduckgo_search_tool()],
            output_type=NativeOutput([Analysis, Unknown]),
        )
        return agent

    def build_web_app(self):
        from fastapi import FastAPI, HTTPException, status

        web_app = FastAPI()
        self.logfire.instrument_fastapi(web_app, capture_headers=True)

        @web_app.post('/webhook')
        async def process_webhook(payload: dict[str, Any]) -> dict[str, Any]:
            if payload['type'] == 'url_verification':
                return {'challenge': payload['challenge']}
            elif (
                payload['type'] == 'event_callback'
                and payload['event']['type'] == 'team_join'
            ):
                profile = Profile.model_validate(payload['event']['user']['profile'])
                self.process_slack_member_in_background(profile)
                return {'status': 'OK'}

            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

        @web_app.get('/analyses')
        async def list_analyses() -> list[Analysis]:
            return await self.store.list()

        return web_app

    def process_slack_member_in_background(self, profile: Profile):
        from logfire.propagate import get_context

        process_slack_member.spawn(profile, logfire_ctx=get_context())

    async def process_slack_member(self, profile: Profile):
        with self.logfire.span('Processing Slack member', profile=profile):
            result = await self.agent.run(profile.as_prompt())
            analysis = result.output
            self.logfire.info('Analysis', analysis=analysis)

            if not isinstance(analysis, Analysis):
                return

            await self.store.add(analysis)

            await self._send_slack_message(
                NEW_LEAD_CHANNEL,
                [
                    {
                        'type': 'header',
                        'text': {
                            'type': 'plain_text',
                            'text': f'New Slack member with score {analysis.relevance}/5',
                        },
                    },
                    {
                        'type': 'divider',
                    },
                    *analysis.as_slack_blocks(),
                ],
            )

    async def send_daily_summary(self):
        with self.logfire.span('Sending daily summary'):
            analyses = await self.store.list()
            self.logfire.info('Analyses', analyses=analyses)

            if len(analyses) == 0:
                return

            sorted_analyses = sorted(analyses, key=lambda x: x.relevance, reverse=True)
            top_analyses = sorted_analyses[:5]

            blocks = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': f'Top {len(top_analyses)} new Slack members from the last 24 hours',
                    },
                },
            ]

            for analysis in top_analyses:
                blocks.extend(
                    [
                        {
                            'type': 'divider',
                        },
                        *analysis.as_slack_blocks(include_relevance=True),
                    ]
                )

            await self._send_slack_message(
                DAILY_SUMMARY_CHANNEL,
                blocks,
            )

            # with self.logfire.span('Clearing analyses'):
            #     await self.store.clear()

    async def _send_slack_message(self, channel: str, blocks: list[dict[str, Any]]):
        import os

        import httpx

        with self.logfire.span('Sending Slack message', channel=channel, blocks=blocks):
            api_key = os.getenv('SLACK_API_KEY', '<unset>')

            client = httpx.AsyncClient()
            response = await client.post(
                'https://slack.com/api/chat.postMessage',
                json={
                    'channel': channel,
                    'blocks': blocks,
                },
                headers={
                    'Authorization': f'Bearer {api_key}',
                },
                timeout=5,
            )
            response.raise_for_status()
            result = response.json()
            if not result.get('ok', False):
                error = result.get('error', 'Unknown error')
                raise Exception(f'Failed to send to Slack: {error}')
