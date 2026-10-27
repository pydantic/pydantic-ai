"""Helpers for the provider batch tests that record real batches."""

from __future__ import annotations

from datetime import timedelta

import anyio
from cassetter import Cassette, RecordMode

from pydantic_ai.models import Model
from pydantic_ai.models.batch import Batch


def batch_poll_interval(vcr: Cassette) -> timedelta:
    """Wait between checks while recording a real batch; replay the recorded checks without waiting."""
    return timedelta(seconds=15) if vcr.record_mode != RecordMode.NONE else timedelta(milliseconds=1)


async def wait_for_batch(model: Model, batch: Batch, vcr: Cassette) -> Batch:
    """Check on `batch` until it has ended, at the pace of `batch_poll_interval`."""
    while batch.status != 'ended':
        await anyio.sleep(batch_poll_interval(vcr).total_seconds())
        batch = await model.batch_status(batch)
    return batch
