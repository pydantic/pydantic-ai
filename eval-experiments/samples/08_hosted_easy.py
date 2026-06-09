"""WITH easy_evals -- same hosted flow. Offline it simulates; with a token it goes live."""

from easy_evals import eval_suite
from easy_evals.hosted import hosted_rubric, push

from fake_models import judge_model, qa_agent

evals = eval_suite(qa_agent(), judge_model=judge_model())
evals.case('Write a haiku about the sea.', judge=hosted_rubric('prompt__haiku_rubric', label='production'))

if __name__ == '__main__':
    push(evals, 'qa-regression')                      # -> Logfire hosted dataset
    evals.run(experiment='nightly-2026-06-09')        # run + stream results to Logfire
