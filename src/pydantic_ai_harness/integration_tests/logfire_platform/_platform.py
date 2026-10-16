"""The Logfire platform's own HTTP APIs, driven the way the Logfire UI drives them.

`AgentControl` never writes a variable: an agent reports its code baseline on an
`agent_control_config_hint` span and creating a config from that is a Logfire-side flow. So a suite
that wants a published config in play has to put one there the way the UI does, and that is what
this module is: `POST /v1/variables/` to create the config, `PUT /v1/variables/<name>/` to publish a
value per label, `DELETE` to remove it, and `GET /v1/query` to read the spans back out.

Two credentials, and they are not interchangeable:

- the **variables API key** (`project:read_variables` + `project:write_variables`) creates and
  publishes. The span write token cannot serve this API.
- the **read token** answers `GET /v1/query`, which is how a test proves a span *arrived* rather
  than that it was *emitted*. Agent Control itself never needs it.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import NoReturn

import httpx
import pytest
from logfire.agent_control import AGENT_CONFIG_JSON_SCHEMA, agent_variable_name
from pydantic import BaseModel, TypeAdapter

AGENT_NAME = f'harness_agent_control_live_{uuid.uuid4().hex}'
"""The agent this suite runs, and so the only variable it ever writes to or deletes.

Unique per run, which is what makes the destructive half of this suite safe rather than merely
documented: every test publishes over this variable and deletes it afterwards, and a name generated
here cannot be the name of a config someone else created. `LOGFIRE_PLATFORM_ALLOW_WRITES` authorizes
writing to a project; it cannot prove the suite owns what is already in one.

A run killed outright (rather than failed, where the fixtures still tear down) can leave one
`agent__harness_agent_control_live_<32 hex>` variable behind. They are safe to delete.
"""

VARIABLE_NAME = agent_variable_name(AGENT_NAME)
"""`agent__harness_agent_control_live_<32 hex>`, derived by the contract's own rule rather than spelled out."""

DEFAULT_TEST_URL = 'http://localhost:3000'
"""Where a local platform stack serves UI, OTLP and API. Override with `LOGFIRE_PLATFORM_TEST_URL`."""

_TIMEOUT = 20.0

_SCHEMA_DOCUMENT = TypeAdapter(dict[str, object])


class SpanRow(BaseModel):
    """One row of a span query: the attributes a test reads its evidence out of."""

    attributes: dict[str, object] = {}


class StoredVariable(BaseModel):
    """The parts of a stored variable definition this suite reads back."""

    kind: str | None = None
    display_name: str | None = None
    description: str | None = None
    example: str | None = None
    json_schema: dict[str, object] | None = None

    def schema_document(self) -> dict[str, object] | None:
        """The stored JSON schema, unwrapped from the `data` envelope the API may return it in."""
        if self.json_schema is None:
            return None
        wrapped = self.json_schema.get('data')
        if wrapped is None:
            return self.json_schema
        return _SCHEMA_DOCUMENT.validate_python(wrapped)


@dataclass(frozen=True)
class Platform:
    """One project's variables and query APIs on a running Logfire platform."""

    base_url: str
    api_key: str
    write_token: str | None
    read_token: str | None

    @property
    def _headers(self) -> dict[str, str]:
        return {'Authorization': f'bearer {self.api_key}', 'Content-Type': 'application/json'}

    def get_variable(self, name: str = VARIABLE_NAME) -> StoredVariable | None:
        """The stored variable definition, or `None` when the project does not have it."""
        response = httpx.get(f'{self.base_url}/v1/variables/{name}/', headers=self._headers, timeout=_TIMEOUT)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return StoredVariable.model_validate(response.json())

    def delete_variable(self, name: str = VARIABLE_NAME, *, attempts: int = 3) -> bool:
        """Delete the variable, reporting whether it was there. Safe to call when it is not.

        Retried on a transport failure, unlike every other call here, because this is the cleanup
        path: what it fails to delete is left behind on someone's project, and a dropped connection
        should not be what leaves it there. A refusal from the server is not retried -- that is an
        answer, and repeating the request would not change it.
        """
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.delete(
                    f'{self.base_url}/v1/variables/{name}/', headers=self._headers, timeout=_TIMEOUT
                )
            except httpx.TransportError:
                if attempt == attempts:
                    raise
                time.sleep(attempt)
                continue
            if response.status_code == 404:
                return False
            response.raise_for_status()
            return True
        raise AssertionError('unreachable: the loop returns or raises on every attempt')

    def create_variable(
        self, *, example: str | None = None, name: str = VARIABLE_NAME, display_name: str = AGENT_NAME
    ) -> StoredVariable:
        """Create the config the way Logfire's promote-a-hint flow has to.

        `display_name` is the agent's name as written, which is what the hint span's
        `agent_control.agent_name` carries. It is required when `kind='agent'`: without it the API answers
        `400 Agent Control variables require a display name`. `json_schema` is the contract's own
        `AGENT_CONFIG_JSON_SCHEMA` rather than one derived from the Pydantic model, because the UI
        edits against it and the platform validates later versions against it. `example` is the
        `agent_control.baseline` off the hint span, verbatim, which is the document the editor shows
        a published value as changes to.
        """
        body: dict[str, object] = {
            'name': name,
            'kind': 'agent',
            'display_name': display_name,
            'description': 'Agent Control conformance suite (pydantic-ai-harness integration tests).',
            'json_schema': AGENT_CONFIG_JSON_SCHEMA,
            'rollout': {'labels': {}},
            'overrides': [],
        }
        if example is not None:
            body['example'] = example
        response = httpx.post(f'{self.base_url}/v1/variables/', headers=self._headers, json=body, timeout=_TIMEOUT)
        response.raise_for_status()
        return StoredVariable.model_validate(response.json())

    def publish(
        self,
        values: dict[str, object],
        *,
        name: str = VARIABLE_NAME,
        rollout: dict[str, float] | None = None,
        serialized: bool = False,
    ) -> None:
        """Publish one value per label, the way saving in the UI does.

        Args:
            values: `{label: config}`, each config a plain `AgentConfig`-shaped dict -- or, with
                `serialized=True`, the literal string to store, which is how a value that is not
                valid JSON gets past this client and into the project.
            name: The variable to write.
            rollout: Label weights. Defaults to all of it on the single label when only one is
                given, which is what makes an `AgentControl()` with no `label` resolve it.
            serialized: Treat `values` as already-serialized strings.

        The whole definition goes on every write because the API offers nothing narrower, and the
        server allocates the version when `version` is omitted.
        """
        existing = self.get_variable(name)
        if existing is None:
            raise RuntimeError(f'{name} does not exist yet; call `create_variable` first.')
        labels = {
            label: {
                'target_type': 'version',
                'serialized_value': value if serialized else json.dumps(value),
            }
            for label, value in values.items()
        }
        if rollout is None:
            rollout = {label: 1.0 for label in values} if len(values) == 1 else {}
        body: dict[str, object] = {
            'name': name,
            'description': existing.description,
            'json_schema': AGENT_CONFIG_JSON_SCHEMA,
            'rollout': {'labels': rollout},
            'overrides': [],
            'labels': labels,
        }
        if existing.example is not None:
            body['example'] = existing.example
        response = httpx.put(
            f'{self.base_url}/v1/variables/{name}/', headers=self._headers, json=body, timeout=_TIMEOUT
        )
        response.raise_for_status()

    def query(self, sql: str, *, limit: int = 20) -> list[SpanRow]:
        """Run one SQL query against the project's spans, as rows.

        Needs the read token rather than the variables key. Gate a test on `require_span_read_back`:
        one that reaches here without a read token has nothing to read with.
        """
        assert self.read_token is not None, 'query needs a read token; gate the test on `require_span_read_back`'
        response = httpx.get(
            f'{self.base_url}/v1/query',
            params={'sql': sql, 'limit': limit, 'json_rows': 'true'},
            headers={'Authorization': f'bearer {self.read_token}'},
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        return [SpanRow.model_validate(row) for row in response.json()['rows']]


def _flag(name: str) -> bool:
    return os.environ.get(name, '').lower() in {'1', 'true', 'yes'}


def requires_live() -> bool:
    """Whether an unusable platform should fail the suite rather than skip it.

    No CI job runs this suite, so the default is to skip. Set `LOGFIRE_PLATFORM_REQUIRE_LIVE` when
    you are running it *as* a gate -- in your own pipeline against a stack you brought up on purpose
    -- so a platform that never came up cannot report green having tested nothing.
    """
    return _flag('LOGFIRE_PLATFORM_REQUIRE_LIVE')


def unavailable(message: str) -> NoReturn:
    """Skip (or, under `LOGFIRE_PLATFORM_REQUIRE_LIVE`, fail) because the platform cannot be used."""
    if requires_live():
        pytest.fail(message)
    pytest.skip(message)


def platform_target() -> Platform:
    """The platform this suite is pointed at, or a skip explaining what is missing.

    Reads its own `LOGFIRE_PLATFORM_*` variables rather than the SDK's `LOGFIRE_API_KEY`, so a
    credential that happens to be in the shell for a real project cannot be the one this suite
    publishes and deletes with. `LOGFIRE_PLATFORM_ALLOW_WRITES` is the deliberate step: without it
    the suite does nothing at all.
    """
    if not _flag('LOGFIRE_PLATFORM_ALLOW_WRITES'):
        unavailable(
            f'this suite creates, publishes over and deletes the {VARIABLE_NAME} variable on the '
            'project it is pointed at, and makes real model requests; set '
            'LOGFIRE_PLATFORM_ALLOW_WRITES=1 to say that is what you want'
        )
    api_key = os.environ.get('LOGFIRE_PLATFORM_API_KEY')
    if not api_key:
        unavailable(
            'no LOGFIRE_PLATFORM_API_KEY; the variables API needs an API key with '
            'project:read_variables and project:write_variables (a span write token cannot serve it)'
        )
    base_url = os.environ.get('LOGFIRE_PLATFORM_TEST_URL', DEFAULT_TEST_URL).rstrip('/')
    platform = Platform(
        base_url=base_url,
        api_key=api_key,
        write_token=os.environ.get('LOGFIRE_PLATFORM_WRITE_TOKEN'),
        read_token=os.environ.get('LOGFIRE_PLATFORM_READ_TOKEN'),
    )
    try:
        platform.get_variable()
    except (httpx.HTTPError, OSError, ValueError) as error:
        # `ValueError` covers the reachable-but-not-a-Logfire case: something answering `200` with a
        # body that is not JSON, or JSON that is not a variable, raises `JSONDecodeError` or
        # pydantic's `ValidationError`, both of which are `ValueError`s. Pointing this suite at the
        # wrong port should skip with the reason, like every other way of not having a platform.
        unavailable(f'no usable Logfire platform at {base_url}: {type(error).__name__}: {error}')
    return platform


def require_span_read_back(platform: Platform) -> None:
    """Skip unless the suite can both send a span to the platform and read it back out.

    Two credentials, two reasons to skip: without the write token the span never leaves the process,
    and without the read token there is nothing to query the platform with.
    """
    if platform.write_token is None:
        pytest.skip('no LOGFIRE_PLATFORM_WRITE_TOKEN; without it no span leaves the process')
    if platform.read_token is None:
        pytest.skip('no LOGFIRE_PLATFORM_READ_TOKEN; the query API is what reads a span back')


def require_provider_key(name: str) -> None:
    """Skip when the provider key a test's model needs is not in the environment."""
    if not os.environ.get(name):
        pytest.skip(f'no {name}; this test makes a real model request')
