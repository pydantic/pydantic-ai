"""WITH easy_evals -- the same several checks, as declarative kwargs (no custom classes)."""

from easy_evals import eval_suite

from fake_models import qa_agent

evals = eval_suite(qa_agent())
evals.case('What is the capital of Japan?', contains='Tokyo', excludes='Paris', max_words=20)

if __name__ == '__main__':
    evals.run()
