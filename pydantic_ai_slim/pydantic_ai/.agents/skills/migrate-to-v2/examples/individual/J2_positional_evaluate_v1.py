"""v1: Dataset.evaluate(task, 'name') with `name` as positional arg is deprecated."""
import asyncio

from pydantic_evals import Case, Dataset


def trigger():
    # DEPRECATION: J2_positional_evaluate
    ds = Dataset(name='x', cases=[Case(name='c', inputs='i', expected_output='o')])

    async def task(i):
        return i

    asyncio.run(ds.evaluate(task, 'positional_name'))


EXPECT = 'positionally to `Dataset.evaluate`'

if __name__ == '__main__':
    trigger()
