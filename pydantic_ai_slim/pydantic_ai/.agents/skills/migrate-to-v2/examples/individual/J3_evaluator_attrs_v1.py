"""v1: setting `evaluation_name` / `evaluator_version` as evaluator class attrs is deprecated."""
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator


@dataclass
class CustomEval(Evaluator):
    evaluation_name: str = 'my_eval_name'
    evaluator_version: str = 'v1'

    def evaluate(self, ctx):
        return True


def trigger():
    # DEPRECATION: J3_evaluator_attrs — fires when the helper methods read the attrs.
    e = CustomEval()
    e.get_default_evaluation_name()
    e.get_evaluator_version()


EXPECT = 'relies on the `evaluation_name` attribute'

if __name__ == '__main__':
    trigger()
