from threading import Thread

from pydantic_graph._utils import get_event_loop, infer_obj_name


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
