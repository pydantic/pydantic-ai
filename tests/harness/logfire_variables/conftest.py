"""Shared fixtures for the `logfire_variables` tests.

These tests rely on the no-provider fallback: resolution uses code defaults
because no Logfire variable provider is configured. Logfire lazily creates a
`LogfireRemoteVariableProvider` that issues requests to the Logfire API
whenever a credential such as `LOGFIRE_API_KEY` is present in `os.environ`,
even with `send_to_logfire=False`. A credential inherited from the developer's
shell, or from a `.env` file loaded into the process, would therefore make
these tests write to the real Logfire API from background publish threads.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

_LOGFIRE_CREDENTIAL_VARS = ('LOGFIRE_TOKEN', 'LOGFIRE_API_KEY')


@pytest.fixture(autouse=True, scope='package')
def _scrub_logfire_credentials() -> Iterator[None]:
    """Remove ambient Logfire credentials while this package's tests run.

    Package scope orders this ahead of the module-scoped `logfire.configure()`
    fixtures in the test modules, which capture `api_key` at configure time.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        for name in _LOGFIRE_CREDENTIAL_VARS:
            monkeypatch.delenv(name, raising=False)
        yield
