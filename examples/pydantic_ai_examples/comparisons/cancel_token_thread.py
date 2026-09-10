"""A stop button, from another thread. CancellationToken interrupts a blocked
run_sync(); the run ends in RunCancelled instead of hanging forever.
"""
import asyncio
import threading
import time
from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse


async def model(messages, info):
    await asyncio.sleep(3600)  # model appears to hang


token = CancellationToken()
agent = Agent(FunctionModel(model))


def stop_handler():
    time.sleep(0.5)
    token.cancel()  # thread-safe: delivered onto the run's loop


def main() -> None:
    stop = threading.Thread(target=stop_handler)
    stop.start()
    try:
        agent.run_sync('go', cancellation_token=token)
        print('BUG: run completed')
    except RunCancelled as exc:
        stop.join()
        print('blocked run_sync interrupted from another thread -> RunCancelled')


if __name__ == '__main__':
    main()
