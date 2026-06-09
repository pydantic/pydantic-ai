"""WITH easy_evals -- plain-English judge criterion."""

from easy_evals import eval_suite

from fake_models import judge_model, qa_agent

evals = eval_suite(qa_agent(), judge_model=judge_model())
evals.case('Write a haiku about the sea.', judge='is a haiku (three lines)')

if __name__ == '__main__':
    evals.run()
