"""v2: pass `name` as a keyword argument to Dataset.evaluate."""
import asyncio

from pydantic_evals import Case, Dataset


def trigger():
    ds = Dataset(name='x', cases=[Case(name='c', inputs='i', expected_output='o')])

    async def task(i):
        return i

    asyncio.run(ds.evaluate(task, name='kwargs_name'))


if __name__ == '__main__':
    trigger()
