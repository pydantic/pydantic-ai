"""v2: override `get_default_evaluation_name` / `get_evaluator_version` instead of setting attrs."""
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator


@dataclass
class CustomEval(Evaluator):
    def evaluate(self, ctx):
        return True

    def get_default_evaluation_name(self) -> str:
        return 'my_eval_name'

    def get_evaluator_version(self) -> str | None:
        return 'v1'


def trigger():
    e = CustomEval()
    e.get_default_evaluation_name()
    e.get_evaluator_version()


if __name__ == '__main__':
    trigger()
