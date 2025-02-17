"""TODO: Remove this comment before merging..

# TODO: Add commit hash, timestamp, and other metadata here (like pytest-speed does), possibly in a dedicated struct
# TODO: Implement serialization of reports for later comparison, and add git hashes etc.
#   Note: I made pydantic_ai.evals.reports.EvalReport a BaseModel specifically to make this easier
# TODO: Implement a CLI with some pytest-like filtering API to make it easier to run only specific cases

# TODO: Add commit hash, timestamp, and other metadata here (like pytest-speed does), possibly in a dedicated struct
"""

# TODO: Make these relative imports; I've used an absolute import for now just to make it possible to run this file directly
from pydantic_ai.evals.evals import evaluation, increment_eval_metric
from pydantic_ai.evals.reports import RenderNumberConfig, RenderValueConfig

__all__ = (
    'evaluation',
    'increment_eval_metric',
    'RenderNumberConfig',
    'RenderValueConfig',
)


async def main():
    # TODO: Remove this before merging
    from functools import partial

    import logfire

    logfire.configure(send_to_logfire=False, console=logfire.ConsoleOptions(verbose=True))

    async def function_i_want_to_evaluate(x: int, deps: str) -> int:
        increment_eval_metric('tokens', len(deps) * x)
        return 2 * x

    with evaluation('my_baseline_eval') as baseline_eval:
        task = partial(function_i_want_to_evaluate, deps='some (non-serializable) dependencies')

        for x in [1, 2, 3]:
            async with baseline_eval.case(task, _case_id=f'{x=}', x=x) as eval_case:
                output = eval_case.case_output
                eval_case.increment_metric('other_metric', 10)
                eval_case.record_score('my_score_1', output / 2)
                eval_case.record_score('my_score_2', output / 10)
                eval_case.record_score('old_score', output / 10)
                eval_case.record_label('sentiment', 'positive' if x == 1 else 'negative')
                eval_case.record_label('old_label', 'hello')

    with evaluation('my_new_eval') as new_eval:
        task = partial(function_i_want_to_evaluate, deps='some other (non-serializable) dependencies')

        for x in [1, 2, 4]:
            async with new_eval.case(task, _case_id=f'{x=}', x=x) as eval_case:
                output = eval_case.case_output
                eval_case.increment_metric('other_metric', 15)
                eval_case.increment_metric('new_metric', 15)
                eval_case.record_score('my_score_1', output / 3)
                eval_case.record_score('my_score_2', output / 6)
                eval_case.record_score('new_score', output + 1)
                eval_case.record_label('sentiment', 'positive')
                eval_case.record_label('new_label', 'world')

    baseline_eval.print_report(include_input=True, include_output=True)
    new_eval.print_report(include_input=True, include_output=True)
    new_eval.print_diff(baseline=baseline_eval, include_input=True, include_output=True, include_removed_cases=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())


# TODO: Use span links to store scores, this provides a way to update them, add them later, etc.
# TODO: Add some kind of `eval_function` decorator, which ensures that calls to the function send eval-review-compatible data to logfire
