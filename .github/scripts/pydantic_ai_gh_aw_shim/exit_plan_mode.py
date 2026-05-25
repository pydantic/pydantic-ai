"""Claude's `ExitPlanMode` tool — acknowledge the end of planning."""

from .shared import logger


def exit_plan_mode(plan: str) -> str:
    """Acknowledge the end of planning.

    The shim has no interactive plan
    review, so this is just a structured ack — the agent continues execution
    against the same workspace it was already operating on.
    """
    logger.info('ExitPlanMode: %s', plan[:200])
    return 'Plan acknowledged — proceeding with execution.'
