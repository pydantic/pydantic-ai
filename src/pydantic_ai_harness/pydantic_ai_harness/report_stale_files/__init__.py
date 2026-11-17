"""Tell the model when files it read have changed in the workspace."""

from pydantic_ai_harness.report_stale_files._capability import STALE_FILES_ORIGIN, ReportStaleFiles

__all__ = [
    'STALE_FILES_ORIGIN',
    'ReportStaleFiles',
]
