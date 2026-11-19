"""Report unhandled tool exceptions to the model instead of ending the run."""

from ._capability import ReportToolErrors, ToolErrorInfo

__all__ = ['ReportToolErrors', 'ToolErrorInfo']
