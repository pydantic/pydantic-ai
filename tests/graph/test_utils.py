from threading import Thread

from pydantic_graph._utils import run_until_complete


def test_run_until_complete_in_main_thread():
    async def run(): ...

    run_until_complete(run())


def test_run_until_complete_in_thread():
    async def run(): ...

    def get_and_close_event_loop():
        run_until_complete(run())

    thread = Thread(target=get_and_close_event_loop)
    thread.start()
    thread.join()
