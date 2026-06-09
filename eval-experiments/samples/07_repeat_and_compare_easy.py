"""WITH easy_evals -- repeat= and baseline= on the suite."""

from easy_evals import eval_suite

from fake_models import qa_agent

evals = eval_suite(qa_agent())
evals.case('What is the capital of France?', expect='Paris')

if __name__ == '__main__':
    baseline = evals.report(repeat=3)
    evals.run(repeat=3, baseline=baseline)
