"""Small GitHub REST client shared by the repository's maintenance scripts."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, cast

_API = 'https://api.github.com'
_RESPONSE_LIMIT = 5_000_000


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a GitHub bearer token through an HTTP redirect."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


class GitHubClient:
    """Small GitHub REST client with bounded response parsing."""

    def __init__(self, token: str) -> None:
        self._token = token

    def _request(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> tuple[Any, str | None]:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f'{_API}{path}',
            data=data,
            method=method,
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self._token}',
                'Content-Type': 'application/json',
                'User-Agent': 'pydantic-ai-maintenance-scripts',
                'X-GitHub-Api-Version': '2022-11-28',
            },
        )
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            if response.status == 204:
                return None, response.headers.get('Link')
            body = response.read(_RESPONSE_LIMIT + 1)
            if len(body) > _RESPONSE_LIMIT:
                raise RuntimeError(f'GitHub response exceeds {_RESPONSE_LIMIT} bytes')
            return json.loads(body), response.headers.get('Link')

    def request(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> Any:
        return self._request(method, path, payload)[0]

    def get(self, path: str) -> Any:
        return self.request('GET', path)

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        return self.request('POST', path, payload)

    def patch(self, path: str, payload: Mapping[str, object]) -> Any:
        return self.request('PATCH', path, payload)

    def delete(self, path: str, payload: Mapping[str, object] | None = None) -> Any:
        return self.request('DELETE', path, payload)

    def last_pages(self, path: str, *, count: int = 1) -> list[dict[str, Any]]:
        """Return up to `count` newest pages for an ascending GitHub collection."""
        separator = '&' if '?' in path else '?'
        first_path = f'{path}{separator}per_page=100&page=1'
        first, links = self._request('GET', first_path)
        last_path = _link_path(links, 'last')
        if not last_path:
            return cast(list[dict[str, Any]], first)
        parsed = urllib.parse.urlparse(last_path)
        query = urllib.parse.parse_qs(parsed.query)
        last = int(query['page'][0])
        pages: list[dict[str, Any]] = []
        for page in range(max(1, last - count + 1), last + 1):
            query['page'] = [str(page)]
            page_path = f'{parsed.path}?{urllib.parse.urlencode(query, doseq=True)}'
            pages.extend(cast(list[dict[str, Any]], self.get(page_path)))
        return pages

    def first_pages(self, path: str, *, count: int) -> tuple[list[dict[str, Any]], bool]:
        """Return up to `count` oldest pages, and whether that was all of them.

        A longer collection is truncated rather than refused, so a huge thread
        costs a bounded prefix instead of aborting the run. The flag lets the
        caller tell "nobody was there" from "we did not get to look".
        """
        separator = '&' if '?' in path else '?'
        page_path = f'{path}{separator}per_page=100&page=1'
        entries: list[dict[str, Any]] = []
        for _ in range(count):
            values, links = self._request('GET', page_path)
            entries.extend(cast(list[dict[str, Any]], values))
            if not (page_path := _link_path(links, 'next')):
                return entries, True
        return entries, False


def _link_path(links: str | None, relation: str) -> str:
    if not links:
        return ''
    for entry in links.split(','):
        if f'rel="{relation}"' in entry:
            url = entry[entry.index('<') + 1 : entry.index('>')]
            parsed = urllib.parse.urlparse(url)
            return f'{parsed.path}?{parsed.query}'
    return ''
