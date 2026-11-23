"""Each Pydantic AI Harness parser of model input raises only the exceptions its contract declares.

See `tests/parser_contracts.py` for the contracts of the core parsers and how they are checked. These call the
parsers directly rather than through an agent run: the inputs are what a model could send, and no recorded exchange
holds most of them.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.file_system import FileSystemToolset
from tests.parser_contracts import CONTRACT_SETTINGS, Call, ParserContract, check, contract

_ROOT = Path(__file__).parent
_FILE_SYSTEM = FileSystemToolset[None](
    root_dir=_ROOT,
    allowed_patterns=[],
    denied_patterns=[],
    max_read_lines=1,
    max_list_results=1,
    max_search_results=1,
    max_find_results=1,
)


def _search_pattern(pattern: str) -> str:
    """Search a directory that doesn't exist, so no search command runs: the pattern is parsed before the path is read.

    Each search gets a loop of its own, leaving alone the current loop the suite gives sync tests.
    """
    search = _FILE_SYSTEM.search_files(pattern, path='no-such-directory', workspace=LocalWorkspaceBackend(_ROOT))
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(search)
    finally:
        loop.close()


CONTRACTS: list[ParserContract] = [
    contract(
        'file-system-search-pattern',
        _search_pattern,
        st.one_of(st.text(), st.from_regex(r'[a(]{0,2}\{\d{1,20}(,\d{0,20})?\}[)+*]{0,2}', fullmatch=True)),
        raises=(ModelRetry,),
        examples=['a{99999999999}', '(a+)+', '['],
    ),
]


@pytest.mark.parametrize(
    ('contract', 'call'),
    [
        pytest.param(contract, call, id=f'{contract.id}-{index}')
        for contract in CONTRACTS
        for index, call in enumerate(contract.examples)
    ],
)
def test_parser_examples_raise_only_what_they_declare(contract: ParserContract, call: Call):
    """The inputs that once raised an undeclared exception now raise only what the contract declares."""
    check(contract, call)


@pytest.mark.parametrize('contract', [pytest.param(contract, id=contract.id) for contract in CONTRACTS])
@CONTRACT_SETTINGS
@given(data=st.data())
def test_parser_inputs_raise_only_what_they_declare(contract: ParserContract, data: st.DataObject):
    check(contract, data.draw(contract.calls))
