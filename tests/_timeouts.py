"""Timeouts shared by tests. It imports nothing, so a module that runs inside a Temporal workflow sandbox can use it."""

READINESS_WAIT_TIMEOUT = 300
"""Bounds a test's wait for something it caused, so only a hang reaches it, however stalled the runner is.

Share this one rather than defining a shorter module constant: a wait that a slow runner can outlast makes the test
depend on runner speed.
"""
