"""v2 form: Dataset(name=..., cases=..., evaluators=...)."""
from pydantic_evals import Dataset, Case


def trigger():
    return Dataset(
        name='my_eval',
        cases=[Case(name='c', inputs='i', expected_output='o')],
        evaluators=[],
    )


if __name__ == '__main__':
    trigger()
