"""WITH easy_evals -- one kwarg; tracing is configured for you."""

from easy_evals import eval_suite

from fake_models import tool_agent

evals = eval_suite(tool_agent())
evals.case('What is the weather in Paris?', expect='sunny', calls_tool='get_weather')

if __name__ == '__main__':
    evals.run()
