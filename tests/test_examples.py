import pytest
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_mock import MockerFixture

LINE_LENGTH = 88


@pytest.mark.parametrize('example', find_examples('docs'), ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample, mocker: MockerFixture):
    # debug(example)
    ruff_ignore: list[str] = ['D']
    if str(example.path).endswith('docs/index.md'):
        ruff_ignore.append('F841')
    eval_example.set_config(ruff_ignore=ruff_ignore, line_length=LINE_LENGTH)
    if eval_example.update_examples:
        eval_example.format(example)
        # eval_example.run_print_update(example)
    else:
        eval_example.lint(example)
        # eval_example.run_print_check(example)

    # eval_example.run(example)
