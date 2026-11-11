"""Have the model review and revise its last answer; adds /refine.

The built-in `refinement` plugin: `/refine [focus]` submits the review instructions of harness
`RefineOutput` as the next prompt, so one review round runs when you ask for it.
"""

from collections.abc import Sequence

from pydantic_ai.messages import ModelResponse
from pydantic_ai_harness.refine_output import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import Plugin


class RefinementPlugin(Plugin):
    """`/refine [focus]`: one review round over the last answer."""

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='refine',
                description='Have the model review and revise its last answer; add words to say what to focus on',
                handler=self._refine,
                raw=True,
            ),
        )

    def _refine(self, args: list[str]) -> str:
        if not any(isinstance(message, ModelResponse) for message in self.host.conversation.messages):
            return 'Nothing to refine: the model has not answered yet.'
        focus = f'\n\nFocus on: {args[0]}' if args else ''
        self.host.submit_prompt(DEFAULT_REFINE_OUTPUT_INSTRUCTIONS + focus)
        return ''
