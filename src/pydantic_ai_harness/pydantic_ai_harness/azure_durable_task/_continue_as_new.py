from pydantic import JsonValue


class ContinueAsNew(BaseException):
    """Hand application-owned continuation input back to the generator orchestrator.

    Raise from `AzureDurableTaskDurability(continue_as_new=...)` to unwind the
    async agent run and its resources. Catch outside `yield from run_durable(...)`,
    call the SDK context's `continue_as_new` with `input`, then return immediately.
    The orchestrator chooses `save_events` and `new_version`.

    Carry messages, usage and usage limits explicitly in this JSON input.
    This signal neither chooses a threshold nor checkpoints coroutine state.
    It inherits `BaseException` so ordinary application error handlers do not
    turn continuation into an agent or tool retry.
    """

    def __init__(self, input: JsonValue) -> None:
        super().__init__(input)
        self.input = input
