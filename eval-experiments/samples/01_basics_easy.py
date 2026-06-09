"""WITH easy_evals -- the same basic Q&A correctness."""

from easy_evals import eval_suite

from fake_models import qa_agent

evals = eval_suite(qa_agent())
evals.case('What is the capital of France?', expect='Paris')
evals.case('What is the capital of Japan?', expect='Tokyo')

if __name__ == '__main__':
    evals.run()
