"""v2 form: pydantic_graph.decision (no .beta)."""


def trigger():
    from pydantic_graph.decision import Decision  # noqa: F401
    return Decision


if __name__ == '__main__':
    trigger()
