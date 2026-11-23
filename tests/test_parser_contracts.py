"""Each parser in `tests/parser_contracts.py` raises only the exceptions its contract declares.

These call the parsers directly rather than through an agent run: the inputs are what a provider, a model or a client
could send, and no recorded exchange holds most of them.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from .parser_contracts import CONTRACT_SETTINGS, CONTRACTS, Call, ParserContract, check


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
