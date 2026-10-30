"""Deprecated import location for `pydantic_ai_harness.dynamic_workflow`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.dynamic_workflow` instead.
"""

from pydantic_ai_harness.dynamic_workflow import (
    DynamicWorkflow,
    DynamicWorkflowToolset,
    WorkflowAgent,
    WorkflowResourceLimits,
)
from pydantic_ai_harness.experimental._warn import warn_moved

warn_moved('dynamic_workflow', 'dynamic_workflow')

__all__ = [
    'DynamicWorkflow',
    'DynamicWorkflowToolset',
    'WorkflowAgent',
    'WorkflowResourceLimits',
]
