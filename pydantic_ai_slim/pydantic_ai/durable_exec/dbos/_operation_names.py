from .._operation import DurableOperationId
from .._operation_names import DurableInvocationName, JournalOperationNamer


class DBOSOperationNamer(JournalOperationNamer):
    """Generate DBOS step names that are persisted compatibility data.

    These names must essentially never change. Changing them can strand in-flight workflows and
    recorded runs.
    """

    def _model_suffix(self, model_id: str | None) -> str:
        return ''

    def invocation_name(self, operation_id: DurableOperationId, *, label: str | None) -> DurableInvocationName:
        return DurableInvocationName(self.operation_name(operation_id))
