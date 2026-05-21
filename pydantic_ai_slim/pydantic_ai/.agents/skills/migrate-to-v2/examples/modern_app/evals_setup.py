"""v2: Dataset with name=."""
from pydantic_evals import Dataset, Case


def build():
    return Dataset(
        name='my_eval',
        cases=[Case(name='c', inputs='i', expected_output='o')],
        evaluators=[],
    )
