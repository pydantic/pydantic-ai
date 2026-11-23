"""Gives each pytest process its own home directory.

Registered by `tests/conftest.py`; see "Home directory" in `tests/AGENTS.md`.

Every xdist worker, and every subprocess it starts, would otherwise share the per-user state that tools keep under the
real home directory: IPython's history database, Jupyter's kernel connection files, `~/.pydantic-ai`, and the config
of every SDK. A kernel in one worker can then move another worker's history file away mid-test. Before collection,
this plugin points `HOME`, the XDG config, data and state directories and `IPYTHONDIR` at a temporary directory of
this process's own, and removes it when the process ends.

Caches stay shared, so downloads such as the Hugging Face models CI restores into `~/.cache` are still found. So do
the credential files under the real home that live recordings need.
"""

from __future__ import annotations as _annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

PLUGIN_NAME = 'pydantic_ai_home_isolation'

_CREDENTIAL_PATHS = {
    'AWS_CONFIG_FILE': '.aws/config',
    'AWS_SHARED_CREDENTIALS_FILE': '.aws/credentials',
    'CLOUDSDK_CONFIG': '.config/gcloud',
    'AZURE_CONFIG_DIR': '.azure',
    'CODEX_HOME': '.codex',
    'DOCKER_CONFIG': '.docker',
    'KUBECONFIG': '.kube/config',
    'MODAL_CONFIG_PATH': '.modal.toml',
}
"""Where each tool looks for its credentials under the home directory, by the variable that overrides it."""

_home: Path | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _home
    real = Path.home()
    pinned = {'XDG_CACHE_HOME': os.environ.get('XDG_CACHE_HOME') or str(real / '.cache')}
    # The Python installs uv manages, which a `uv run` in a test would otherwise download again.
    uv_pythons = Path(os.environ.get('XDG_DATA_HOME') or real / '.local' / 'share') / 'uv' / 'python'
    if 'UV_PYTHON_INSTALL_DIR' not in os.environ and uv_pythons.exists():  # pragma: lax no cover
        pinned['UV_PYTHON_INSTALL_DIR'] = str(uv_pythons)
    for variable, path in _CREDENTIAL_PATHS.items():
        if variable not in os.environ and (real / path).exists():  # pragma: lax no cover
            pinned[variable] = str(real / path)
    _home = Path(tempfile.mkdtemp(prefix='pydantic-ai-home-'))
    isolated = {
        'HOME': str(_home),
        'XDG_CONFIG_HOME': str(_home / '.config'),
        'XDG_DATA_HOME': str(_home / '.local' / 'share'),
        'XDG_STATE_HOME': str(_home / '.local' / 'state'),
        'IPYTHONDIR': str(_home / '.ipython'),
    }
    # Created up front: a tool that creates its directory on first use, such as IPython, would otherwise do it in a test's
    # event loop, where `blockbuster` fails the blocking call.
    for directory in isolated.values():
        Path(directory).mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':  # pragma: lax no cover
        isolated['USERPROFILE'] = str(_home)
    os.environ.update(pinned | isolated)


def pytest_unconfigure(config: pytest.Config) -> None:
    if _home is not None:  # pragma: no branch
        shutil.rmtree(_home, ignore_errors=True)
