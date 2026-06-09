"""WITH easy_evals -- partial field match via expect={...}."""

from easy_evals import eval_suite

from fake_models import person_agent

evals = eval_suite(person_agent())
evals.case('Make a person named Ada aged 36', expect={'name': 'Ada', 'age': 36})

if __name__ == '__main__':
    evals.run()
