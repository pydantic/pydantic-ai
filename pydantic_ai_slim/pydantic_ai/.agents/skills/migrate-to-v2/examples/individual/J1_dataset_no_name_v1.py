"""v1: Dataset() without name=."""
from pydantic_evals import Dataset, Case


def trigger():
    # DEPRECATION: J1_dataset_no_name
    return Dataset(cases=[Case(name='c', inputs='i', expected_output='o')], evaluators=[])


EXPECT = 'Omitting the `name` parameter is deprecated'

if __name__ == '__main__':
    trigger()
