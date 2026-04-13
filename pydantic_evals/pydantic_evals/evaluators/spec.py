"""This module provides the EvaluatorSpec class for specifying evaluators in a serializable format.

EvaluatorSpec is a type alias for `NamedSpec` from the shared `pydantic_ai._spec` module.
"""

from pydantic_ai._spec import NamedSpec

EvaluatorSpec = NamedSpec
"""The specification of an evaluator to be run.

This class is used to represent evaluators in a serializable format, supporting various
short forms for convenience when defining evaluators in YAML or JSON dataset files.

In particular, each of the following forms is supported for specifying an evaluator with name `MyEvaluator`:
* `'MyEvaluator'` - Just the (string) name of the Evaluator subclass is used if its `__init__` takes no arguments
* `{'MyEvaluator': first_arg}` - A single argument is passed as the first positional argument to `MyEvaluator.__init__`
* `{'MyEvaluator': {k1: v1, k2: v2}}` - Multiple kwargs are passed to `MyEvaluator.__init__`
"""
