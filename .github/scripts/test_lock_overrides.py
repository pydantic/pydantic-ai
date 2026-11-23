from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from lock_overrides import expired, main, remove, timestamp

# The override table on the parent of https://github.com/pydantic/pydantic-ai/pull/13799, which removed by hand every
# override that had expired by the time it merged, 8 to 25 days after each expired.
BEFORE_13799 = """\
[tool.uv]
# `exclude-newer` relative durations need >= 0.9.17, `exclude-newer-package` `false` needs >= 0.9.25
required-version = ">=0.9.25"
exclude-newer = "7 days"

[tool.uv.exclude-newer-package]
# These are our own packages, not worth defending from supply chain attacks on them.
# `listentome` and `cassetter` are maintained by a Pydantic team member and count as ours.
listentome = false
cassetter = false
pydantic = false
pydantic-core = false
pydantic-extra-types = false
pydantic-settings = false
pydantic-handlebars = false
pydantic-monty = false
pydantic-monty-client = false
pydantic-monty-runtime = false
genai-prices = false
pydantic-ai-harness = false
logfire = false
logfire-api = false
# certifi 2026.10.19 is the `certifi` floor above: it keeps out the compromised eMudhra
# `emSign ECC Root CA - G3` and restores the unaffected emSign roots 2026.10.18 dropped.
# TODO(dsfaccini): Remove after 2026-10-27 02:25 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
certifi = "2026-10-20T02:25:00Z"
# pillow 13.0.0 is the `pillow` floor above: it is the first release with the fixes.
# TODO(dsfaccini): Remove after 2026-10-23 22:19 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
pillow = "2026-10-16T22:19:00Z"
# boxlite 0.10.7 is the `boxlite` floor: 0.10.6 has no wheels for Python 3.13 or 3.14.
# TODO(adtyavrdhn): Remove after 2026-10-25 01:35 UTC.
boxlite = "2026-10-18T01:35:01Z"
# prefect 3.9.0 is the `prefect` floor: it is the first release that imports on a current FastAPI.
# TODO(DouweM): Remove after 2026-10-28 03:46 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
prefect = "2026-10-21T03:46:00Z"
# boxd 0.2.20 is the `boxd` floor: it is the first release with `machines.fs`.
# TODO(adtyavrdhn): Remove after 2026-11-03 23:56 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
boxd = "2026-10-27T23:56:00Z"
# render 1.6.0 provides the stable Sandbox API used by RenderSandbox.
# TODO(adtyavrdhn): Remove after 2026-11-06 01:06 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
render = "2026-10-30T01:06:03Z"
# k8s-agent-sandbox 1.0.11 is the `kubernetes` extra's floor: the release the backend is verified against.
# TODO(adtyavrdhn): Remove after 2026-11-19 19:37 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
k8s-agent-sandbox = "2026-11-12T19:37:00Z"
# durabletask and durabletask-azuremanaged 1.13.0 are the floors for the `azure-durable-task` extra
# of pydantic-ai-harness: they are the first releases that close a dispatch's orchestrator generator.
# TODO(adtyavrdhn): Remove after 2026-11-09 22:34 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
durabletask = "2026-11-02T22:34:00Z"
durabletask-azuremanaged = "2026-11-02T22:34:00Z"

[dependency-groups]
dev = [
    "anyio>=4.11.0",
]
"""

AFTER_13799 = """\
[tool.uv]
# `exclude-newer` relative durations need >= 0.9.17, `exclude-newer-package` `false` needs >= 0.9.25
required-version = ">=0.9.25"
exclude-newer = "7 days"

[tool.uv.exclude-newer-package]
# These are our own packages, not worth defending from supply chain attacks on them.
# `listentome` and `cassetter` are maintained by a Pydantic team member and count as ours.
listentome = false
cassetter = false
pydantic = false
pydantic-core = false
pydantic-extra-types = false
pydantic-settings = false
pydantic-handlebars = false
pydantic-monty = false
pydantic-monty-client = false
pydantic-monty-runtime = false
genai-prices = false
pydantic-ai-harness = false
logfire = false
logfire-api = false
# k8s-agent-sandbox 1.0.11 is the `kubernetes` extra's floor: the release the backend is verified against.
# TODO(adtyavrdhn): Remove after 2026-11-19 19:37 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
k8s-agent-sandbox = "2026-11-12T19:37:00Z"

[dependency-groups]
dev = [
    "anyio>=4.11.0",
]
"""

MERGED_13799 = '2026-11-18T03:07:33Z'
REMOVED_BY_13799 = [
    'certifi',
    'pillow',
    'boxlite',
    'prefect',
    'boxd',
    'render',
    'durabletask',
    'durabletask-azuremanaged',
]


def test_lists_the_overrides_13799_removed():
    """At the merge of the PR that removed them by hand, exactly those overrides are expired."""
    assert expired(BEFORE_13799, timestamp(MERGED_13799)) == REMOVED_BY_13799


def test_remove_deletes_the_expired_entries_and_their_comments_and_nothing_else():
    """The table reads as that PR left it, and every line outside the removed blocks is unchanged."""
    assert remove(BEFORE_13799, REMOVED_BY_13799) == AFTER_13799


@pytest.mark.parametrize(
    ('value', 'now', 'is_expired'),
    [
        # The TODO comment above the entry says to remove it after 2026-11-19 19:37 UTC.
        pytest.param('2026-11-17T00:00:00Z', '2026-11-23T00:00:00Z', False, id='comment-past-value-in-window'),
        pytest.param('2026-11-01T00:00:00Z', '2026-11-19T00:00:00Z', True, id='comment-future-value-behind'),
    ],
)
def test_expiry_reads_the_value_not_the_comment(value: str, now: str, is_expired: bool):
    """Only the override's date decides, whatever its TODO comment says."""
    text = BEFORE_13799.replace('k8s-agent-sandbox = "2026-11-12T19:37:00Z"', f'k8s-agent-sandbox = "{value}"')
    assert ('k8s-agent-sandbox' in expired(text, timestamp(now))) is is_expired


@pytest.mark.parametrize(
    ('now', 'is_expired'),
    [
        pytest.param('2026-11-19T19:36:59Z', False, id='inside-the-window'),
        pytest.param('2026-11-19T19:37:00Z', False, id='at-the-window-edge'),
        pytest.param('2026-11-19T19:37:01Z', True, id='behind-the-window'),
    ],
)
def test_expires_once_the_value_is_older_than_now_minus_the_span(now: str, is_expired: bool):
    """`k8s-agent-sandbox = "2026-11-12T19:37:00Z"` under a 7-day span expires after 2026-11-19 19:37 UTC."""
    assert ('k8s-agent-sandbox' in expired(BEFORE_13799, timestamp(now))) is is_expired


def test_false_overrides_never_expire():
    """However late it is, only the dated overrides are expired."""
    assert expired(BEFORE_13799, timestamp('2100-01-01T00:00:00Z')) == [
        *REMOVED_BY_13799[:6],
        'k8s-agent-sandbox',
        *REMOVED_BY_13799[6:],
    ]


def test_a_shared_comment_stays_while_an_entry_under_it_stays():
    """Removing one of two entries under one comment block keeps the block for the other."""
    removed = remove(BEFORE_13799, ['durabletask'])
    assert removed == BEFORE_13799.replace('durabletask = "2026-11-02T22:34:00Z"\n', '')
    assert remove(removed, ['durabletask-azuremanaged']) == BEFORE_13799.replace(
        """\
# durabletask and durabletask-azuremanaged 1.13.0 are the floors for the `azure-durable-task` extra
# of pydantic-ai-harness: they are the first releases that close a dispatch's orchestrator generator.
# TODO(adtyavrdhn): Remove after 2026-11-09 22:34 UTC, then run `uv lock` and
# `uv lock --script .github/scripts/pydantic-ai-runner`.
durabletask = "2026-11-02T22:34:00Z"
durabletask-azuremanaged = "2026-11-02T22:34:00Z"
""",
        '',
    )


def test_a_global_span_it_cannot_read_fails():
    """An `exclude-newer` the script cannot turn into a span fails instead of expiring nothing."""
    text = BEFORE_13799.replace('exclude-newer = "7 days"', 'exclude-newer = "2026-11-01T00:00:00Z"')
    with pytest.raises(ValueError, match='not a relative span'):
        expired(text, timestamp(MERGED_13799))


@pytest.mark.parametrize('newline', ['\n', '\r\n'])
def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str], newline: str):
    """`list` prints the expired names and leaves the file alone; `remove` prints them and rewrites the file."""
    path = tmp_path / 'pyproject.toml'
    path.write_bytes(BEFORE_13799.replace('\n', newline).encode())

    main(['list', '--now', MERGED_13799, str(path)])
    assert capsys.readouterr().out.splitlines() == REMOVED_BY_13799
    assert path.read_bytes() == BEFORE_13799.replace('\n', newline).encode()

    main(['remove', '--now', MERGED_13799, str(path)])
    assert capsys.readouterr().out.splitlines() == REMOVED_BY_13799
    assert path.read_bytes() == AFTER_13799.replace('\n', newline).encode()

    main(['remove', '--now', MERGED_13799, str(path)])
    assert capsys.readouterr().out == ''
