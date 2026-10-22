"""The GitHub REST calls the review gates make.

Everything a pull request contributes arrives here as data: its text, its file list and the contents of
its files at its head. Nothing fetched is ever written to disk, imported or executed.
"""

from __future__ import annotations

import urllib.parse
from typing import Literal

import httpx2
from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict


class _Label(BaseModel):
    name: str


class _Repo(BaseModel):
    full_name: str


class _Head(BaseModel):
    sha: str
    ref: str
    # `null` once the fork is deleted.
    repo: _Repo | None


class _Base(BaseModel):
    ref: str


class PullRequest(BaseModel):
    """A pull request, as the gates read it."""

    number: int
    state: str
    title: str
    body: str | None = None
    labels: list[_Label] = []
    head: _Head
    base: _Base


class ChangedFile(BaseModel):
    """One entry of the compare API's file list."""

    filename: str
    status: str
    # Absent for a binary file or a diff too large for the API to include.
    patch: str | None = None
    previous_filename: str | None = None


class _Commit(BaseModel):
    sha: str


class Comparison(BaseModel):
    """What `head` changes since its merge base with the base branch."""

    merge_base_commit: _Commit
    files: list[ChangedFile] = []


class Issue(BaseModel):
    """A linked issue, read as data."""

    number: int
    title: str
    body: str | None = None


class CheckRun(BaseModel):
    """An existing check run, by its id and the external id that dedupes it."""

    id: int
    external_id: str | None = None


class _CheckRuns(BaseModel):
    check_runs: list[CheckRun]


class Annotation(TypedDict):
    """One check-run annotation."""

    path: str
    start_line: int
    end_line: int
    annotation_level: Literal['notice', 'warning']
    title: str
    message: str


class CheckOutput(TypedDict):
    """A check run's output."""

    title: str
    summary: str
    annotations: list[Annotation]


Conclusion = Literal['success', 'neutral', 'skipped']

# The compare API lists at most this many files; a longer list is cut off without saying so.
COMPARE_FILE_LIMIT = 300

_PULL_REQUESTS = TypeAdapter(list[PullRequest])


class GitHub:
    """The repository the gates judge, read and written with the workflow's token."""

    def __init__(self, client: httpx2.AsyncClient, repo: str) -> None:
        self._client = client
        self._repo = f'/repos/{repo}'

    @classmethod
    def connect(cls, repo: str, token: str) -> GitHub:
        client = httpx2.AsyncClient(
            base_url='https://api.github.com',
            headers={
                'Authorization': f'Bearer {token}',
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
            },
            timeout=30,
        )
        return cls(client, repo)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, str] | None = None) -> httpx2.Response:
        response = await self._client.get(f'{self._repo}/{path}', params=params)
        response.raise_for_status()
        return response

    async def pull_request(self, number: int) -> PullRequest:
        return PullRequest.model_validate_json((await self._get(f'pulls/{number}')).content)

    async def open_pull_requests(self, head: str) -> list[PullRequest]:
        """The open pull requests whose head is `head`, as `owner:branch`."""
        response = await self._get('pulls', {'state': 'open', 'head': head})
        return _PULL_REQUESTS.validate_json(response.content)

    async def comparison(self, base: str, head: str) -> Comparison:
        """The files `head` changes since its merge base with `base`: what the PR's diff shows."""
        return Comparison.model_validate_json((await self._get(f'compare/{base}...{head}')).content)

    async def file_text(self, path: str, ref: str) -> str | None:
        """The text of `path` at `ref`, or `None` when it is absent there."""
        response = await self._client.get(
            f'{self._repo}/contents/{urllib.parse.quote(path)}',
            params={'ref': ref},
            headers={'Accept': 'application/vnd.github.raw+json'},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text

    async def issue(self, number: int) -> Issue | None:
        response = await self._client.get(f'{self._repo}/issues/{number}')
        if response.status_code in (404, 410):
            return None
        response.raise_for_status()
        return Issue.model_validate_json(response.content)

    async def check_runs(self, sha: str, name: str) -> list[CheckRun]:
        response = await self._get(f'commits/{sha}/check-runs', {'check_name': name, 'filter': 'all'})
        return _CheckRuns.model_validate_json(response.content).check_runs

    async def start_check_run(self, *, name: str, head_sha: str, external_id: str) -> int:
        response = await self._client.post(
            f'{self._repo}/check-runs',
            json={'name': name, 'head_sha': head_sha, 'external_id': external_id, 'status': 'in_progress'},
        )
        response.raise_for_status()
        return CheckRun.model_validate_json(response.content).id

    async def finish_check_run(self, check_run_id: int, *, conclusion: Conclusion, output: CheckOutput) -> None:
        response = await self._client.patch(
            f'{self._repo}/check-runs/{check_run_id}',
            json={'status': 'completed', 'conclusion': conclusion, 'output': output},
        )
        response.raise_for_status()
