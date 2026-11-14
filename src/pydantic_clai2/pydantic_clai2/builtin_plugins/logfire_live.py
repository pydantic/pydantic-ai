"""The `observability` plugin's side panel: Logfire's live view of the conversation's trace, in the web UI.

Every run of a conversation nests under its `CLAI session` root, so the root's trace is the
conversation, and the live view filtered to it (`?q=trace_id='...'`, the link Logfire's own MCP
server builds) fills in as the agent works. Once a run started, `traceId` and `spanId` open that
run's details. The project's address comes from Logfire, which names it for the write token.
"""

from urllib.parse import quote, urlencode, urlsplit

import httpx
from opentelemetry.trace import SpanContext, format_span_id, format_trace_id
from pydantic import BaseModel, ValidationError

from pydantic_clai2.builtin_plugins.logfire_setup import Setup


class _Info(BaseModel):
    project_url: str


async def project_url(setup: Setup, *, base_url: str, token: str) -> str | None:
    """The page of the project `token` writes to, as Logfire's `/v1/info` names it; `None` when it does not say."""
    try:
        async with setup.http() as http:
            response = await http.get(f'{base_url.rstrip("/")}/v1/info', headers={'Authorization': token})
            response.raise_for_status()
        found = _Info.model_validate_json(response.content).project_url
    except (httpx.HTTPError, ValidationError):
        return None
    # Only a web page: the browser opens it.
    return found if urlsplit(found).scheme in ('http', 'https') else None


def live_view_url(project: str, *, trace: SpanContext, run: SpanContext | None) -> str:
    """The project's live view showing only `trace`, with `run`'s details open when given."""
    trace_id = format_trace_id(trace.trace_id)
    query = {'q': f"trace_id='{trace_id}'"}
    if run is not None:
        query |= {'traceId': trace_id, 'spanId': format_span_id(run.span_id)}
    return f'{project.rstrip("/")}?{urlencode(query, quote_via=quote)}'
