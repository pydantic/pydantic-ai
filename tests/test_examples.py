import pytest
from pytest_examples import CodeExample, EvalExample, find_examples
from pytest_mock import MockerFixture


@pytest.mark.parametrize('example', find_examples('docs'), ids=str)
def test_docs_examples(example: CodeExample, eval_example: EvalExample, mocker: MockerFixture):
    # debug(example)
    eval_example.lint(example)
    eval_example.run(example)
