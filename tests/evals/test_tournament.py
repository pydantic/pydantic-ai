from __future__ import annotations as _annotations

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.tournament import EvalPlayer


    def test_evalplayer() -> None:
        """
        Test the EvalPlayer class.
        """
        player = EvalPlayer(
            idx=42,
            item='toasted rice & miso caramel ice cream',
        )
        assert player.idx == 42
        assert player.item == 'toasted rice & miso caramel ice cream'

