"""Every model class whose `request_stream` calls its SDK has its streams checked by a parity driver, or says why not.

`STREAM_PARITY` in `tests/models/stream_parity.py` names, for each such class, the stream parity module that replays
its recorded traffic, whole and re-chunked, or the reason none does. A new class that streams through its SDK fails
here until it has an entry.
"""

from __future__ import annotations

import importlib

from ..manifests import assert_classified
from ..model_adapter_cases import sdk_cells, shipped_model_classes
from .stream_parity import STREAM_PARITY, ReplayedBy, StreamParityDriver


def test_every_streaming_cell_names_a_parity_driver() -> None:
    _, skipped = shipped_model_classes()
    assert_classified(
        (cell.model_class.__name__ for cell in sdk_cells() if cell.entry_point == 'request_stream'),
        STREAM_PARITY,
        # Only a `Model` streams, so a skipped embedding or image module hides no entry's member.
        complete_walk=not any(module.startswith('pydantic_ai.models.') for module in skipped),
        table_name='STREAM_PARITY',
    )


def test_every_named_parity_module_has_a_driver() -> None:
    """The table names each driver by its module, which imports the table, so this checks the module defines one."""
    modules = {entry.module for entry in STREAM_PARITY.values() if isinstance(entry, ReplayedBy)}
    missing = sorted(
        module
        for module in modules
        if not isinstance(vars(importlib.import_module(module)).get('DRIVER'), StreamParityDriver)
    )
    assert not missing, f'these modules define no `DRIVER`: {missing}'
