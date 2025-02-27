from threading import Thread

from pydantic_graph._utils import get_event_loop


def test_get_event_loop_in_thread():
    thread = Thread(target=get_event_loop)
    thread.start()
