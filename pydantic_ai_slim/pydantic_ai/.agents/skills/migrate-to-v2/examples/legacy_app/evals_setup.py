"""v1: Dataset without name=."""
from pydantic_evals import Dataset, Case


def build():
    # DEPRECATION: J1_dataset_no_name
    return Dataset(
        cases=[Case(name='c', inputs='i', expected_output='o')],
        evaluators=[],
    )
