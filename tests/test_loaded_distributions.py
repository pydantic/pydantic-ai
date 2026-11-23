import pytest
from pydantic import __file__ as pydantic_file

from tests.loaded_distributions import assert_loads_only


def test_a_distribution_off_the_allowlist_fails():
    """An entry point that newly loads a distribution fails, naming it and the list to add it to."""
    with pytest.raises(AssertionError, match=r"^startup now loads \['pydantic'\]\. .* add each to `ALLOWED` "):
        assert_loads_only('startup', {'pydantic': pydantic_file}, {}, table_name='ALLOWED')
