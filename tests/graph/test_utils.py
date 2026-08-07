from collections.abc import Generator
from threading import Thread
from typing import Any

import pytest

from pydantic_graph._utils import get_event_loop, infer_obj_name, run_until_complete
from pydantic_graph.exceptions import UnsupportedEventLoopError

from .._inline_snapshot import snapshot
from ..conftest import undrivable_event_loop


def test_get_event_loop_in_thread():
    def get_and_close_event_loop():
        event_loop = get_event_loop()
        event_loop.close()

    thread = Thread(target=get_and_close_event_loop)
    thread.start()
    thread.join()


def test_infer_obj_name():
    """Test inferring variable names from the calling frame."""
    my_object = object()
    # Depth 1 means we look at the frame calling infer_obj_name
    inferred = infer_obj_name(my_object, depth=1)
    assert inferred == 'my_object'

    # Test with object not in locals
    result = infer_obj_name(object(), depth=1)
    assert result is None


def test_infer_obj_name_no_frame():
    """Test infer_obj_name when frame inspection fails."""
    # This is hard to trigger without mocking, but we can test that the function
    # returns None gracefully when it can't find the object
    some_obj = object()

    # Call with depth that would exceed the call stack
    result = infer_obj_name(some_obj, depth=1000)
    assert result is None


global_obj = object()


def test_infer_obj_name_locals_vs_globals():
    """Test infer_obj_name prefers locals over globals."""
    result = infer_obj_name(global_obj, depth=1)
    assert result == 'global_obj'

    # Assign a local name to the variable and ensure it is found with precedence over the global
    local_obj = global_obj
    result = infer_obj_name(global_obj, depth=1)
    assert result == 'local_obj'

    # If we unbind the local name, should find the global name again
    del local_obj
    result = infer_obj_name(global_obj, depth=1)
    assert result == 'global_obj'


def test_run_until_complete_on_undrivable_event_loop():
    """An event loop that doesn't implement `run_until_complete()` is reported before anything is scheduled.

    Temporal's workflow event loop is like this, and the bare `NotImplementedError` CPython raises for it
    isn't a type Temporal's durable execution integration recognizes, so it retries the workflow task forever
    instead of failing the workflow. See https://github.com/pydantic/pydantic-ai/issues/6899.
    """
    started = False

    async def coro() -> None:
        nonlocal started
        started = True  # pragma: no cover

    with undrivable_event_loop():
        with pytest.raises(UnsupportedEventLoopError) as exc_info:
            run_until_complete(coro())

    assert str(exc_info.value) == snapshot(
        'The current event loop (UndrivableEventLoop) does not implement `run_until_complete()`, which synchronous methods need in order to run their asynchronous implementation. This is the case inside a Temporal workflow, whose event loop can only be driven by Temporal itself. Use the asynchronous method instead, e.g. `await agent.run()` rather than `agent.run_sync()`.'
    )
    assert not started


def test_run_until_complete_on_undrivable_event_loop_with_non_coroutine_awaitable():
    """Awaitables that aren't coroutines (and so can't be closed) are handled just the same."""

    class NonCoroutineAwaitable:
        def __await__(self) -> Generator[Any, Any, None]:
            yield  # pragma: no cover

    with undrivable_event_loop():
        with pytest.raises(UnsupportedEventLoopError):
            run_until_complete(NonCoroutineAwaitable())


def test_run_until_complete_propagates_not_implemented_error_from_coroutine():
    """A `NotImplementedError` raised by the coroutine itself must not be mistaken for an unsupported loop.

    This is why the loop is checked up front instead of by catching `NotImplementedError` around the call:
    user code raising `NotImplementedError` (an abstract method, a stub, a tool function) has nothing to do
    with the event loop.
    """

    async def coro() -> None:
        raise NotImplementedError('Not implemented by the user')

    with pytest.raises(NotImplementedError) as exc_info:
        run_until_complete(coro())

    assert type(exc_info.value) is NotImplementedError
    assert str(exc_info.value) == snapshot('Not implemented by the user')


def test_graph_exceptions():
    """Construct each public graph exception to assert their `__init__`s wire `message` and the underlying class."""
    from pydantic_graph.exceptions import GraphRuntimeError, GraphSetupError

    setup_err = GraphSetupError('bad node')
    assert setup_err.message == 'bad node'
    assert str(setup_err) == 'bad node'
    assert isinstance(setup_err, TypeError)

    runtime_err = GraphRuntimeError('bad run')
    assert runtime_err.message == 'bad run'
    assert str(runtime_err) == 'bad run'
    assert isinstance(runtime_err, RuntimeError)
