"""Stand-ins for the remote clients and executables capability tools call, each counting the requests it receives.

The cases in `tests/harness/capability_cases.py` build capabilities with these so their tools run offline. The
SDK types they return are imported where they are built, so a lane without `exa-py` or `youdotcom` can still
import this module and build the cases that need neither.
"""

from __future__ import annotations

import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

if TYPE_CHECKING:
    from exa_py.agent.types import AgentEffort, AgentRun
    from exa_py.api import (
        ContentsOptions,
        DeepOutputSchema,
        Result,
        SearchResponse,
        SearchType,
        TextContentsOptions,
    )
    from youdotcom import models

REQUESTS: Counter[str] = Counter()
"""Requests the stand-ins received, by request name."""


class ExaClient:
    async def search(
        self,
        query: str,
        *,
        contents: ContentsOptions | Literal[False],
        num_results: int | None = None,
        type: SearchType | None = None,
        output_schema: DeepOutputSchema | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> SearchResponse[Result]:
        from exa_py.api import DeepSearchOutput, Result, SearchResponse

        REQUESTS[f'exa.search.{type}'] += 1
        output = DeepSearchOutput(content='deep answer', grounding=[]) if type == 'deep' else None
        result = Result(url='https://a.dev', id='a', title='A', highlights=['excerpt'])
        return SearchResponse(results=[result], resolved_search_type=None, auto_date=None, output=output)

    async def get_contents(self, urls: str, *, text: TextContentsOptions) -> SearchResponse[Result]:
        from exa_py.api import Result, SearchResponse

        REQUESTS['exa.get_contents'] += 1
        result = Result(url=urls, id=urls, title='A', text='page text')
        return SearchResponse(results=[result], resolved_search_type=None, auto_date=None)


class ExaAgentRuns:
    async def create(
        self,
        *,
        query: str,
        system_prompt: str | None = None,
        output_schema: dict[str, object] | type[BaseModel] | None = None,
        effort: AgentEffort | None = None,
        previous_run_id: str | None = None,
    ) -> AgentRun:
        from exa_py.agent.types import AgentRun

        REQUESTS['exa_agent.create'] += 1
        return AgentRun(id='run_1', status='queued')

    async def poll_until_finished(
        self, run_id: str, *, poll_interval: int = 1000, timeout_ms: int = 3600000
    ) -> AgentRun:
        from exa_py.agent.types import AgentOutput, AgentRun

        REQUESTS['exa_agent.poll'] += 1
        return AgentRun(id=run_id, status='completed', output=AgentOutput(text='Done.'))


class YouClient:
    async def search_async(
        self,
        *,
        query: str,
        count: int | None = None,
        freshness: str | None = None,
        country: str | None = None,
        extraction: models.Extraction | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.SearchResponse:
        from youdotcom import models

        REQUESTS['you.search'] += 1
        return models.SearchResponse()

    async def contents_async(
        self,
        *,
        urls: Sequence[str] | None = None,
        formats: Sequence[models.ContentsFormats] | None = None,
    ) -> list[models.ContentsResponse]:
        from youdotcom import models

        REQUESTS['you.contents'] += 1
        return [models.ContentsResponse(url='https://a.dev', title='A', markdown='body')]

    async def answer_async(
        self,
        *,
        query: str,
        freshness: str | None = None,
        country: str | None = None,
        include_domains: Sequence[str] | None = None,
        exclude_domains: Sequence[str] | None = None,
        boost_domains: Sequence[str] | None = None,
    ) -> models.AnswerResponse:
        from youdotcom import models

        REQUESTS['you.answer'] += 1
        return models.AnswerResponse(answer='the answer')

    async def research_async(
        self,
        *,
        input: str,
        research_effort: models.ResearchEffort | None = None,
        background: bool | None = None,
        source_control: models.SourceControl | Mapping[str, object] | None = None,
        output_schema: Mapping[str, object] | None = None,
    ) -> models.ResearchResult:
        from youdotcom import models

        REQUESTS['you.research'] += 1
        # An empty answer makes the tool ask the model to retry, which the operation records as data.
        return models.ResearchResponse(
            output=models.Output(content='', content_type=models.ContentType.TEXT, sources=[]), warnings=None
        )

    async def finance_research_async(
        self,
        *,
        input: str,
        research_effort: models.FinanceResearchEffort | None = None,
    ) -> models.FinanceResearchResponse:
        from youdotcom import models

        REQUESTS['you.finance_research'] += 1
        return models.FinanceResearchResponse(
            output=models.FinanceResearchOutput(
                content='finance answer', content_type=models.FinanceResearchContentType.TEXT, sources=[]
            )
        )


class KeenableClient:
    async def search(self, query: str, *, max_results: int) -> list[dict[str, object]]:
        REQUESTS['keenable.search'] += 1
        return [{'url': 'https://a.dev', 'title': 'A', 'snippet': 'excerpt'}]

    async def fetch(self, url: str) -> dict[str, object]:
        REQUESTS['keenable.fetch'] += 1
        return {'url': url, 'title': 'A', 'content': 'body'}


AWS_CLI_COUNT_VARIABLE = 'HARNESS_AWS_CLI_COUNT'
"""The environment variable naming the file the `aws` stand-in appends a line to per run, when it is set."""


def counting_aws_cli(directory: Path) -> str:
    """An `aws` stand-in in `directory`, which prints `ok` and counts its runs."""
    script = directory / 'aws'
    script.write_text(
        f'#!/bin/sh\n[ -n "${AWS_CLI_COUNT_VARIABLE}" ] && echo run >> "${AWS_CLI_COUNT_VARIABLE}"\necho ok\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def counting_model(name: str, respond: Callable[[AgentInfo], ModelResponse]) -> FunctionModel:
    """A model that answers with `respond` and counts its requests under `name`."""

    def counted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        REQUESTS[name] += 1
        return respond(info)

    return FunctionModel(counted)


def judge_relevant(info: AgentInfo) -> ModelResponse:
    """A `SmartSearch` judge's verdict that every candidate is relevant."""
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'entity': 1, 'operation': 1})])
