from threading import Thread

from pydantic_graph._utils import get_event_loop


def test_get_event_loop_in_thread():
    def get_and_close_event_loop():
        event_loop = get_event_loop()
        event_loop.close()

    thread = Thread(target=get_and_close_event_loop)
    thread.start()
    thread.join()
