"""v1: pydantic_graph.beta.decision import."""


def trigger():
    # DEPRECATION: I2_graph_beta
    from pydantic_graph.beta.decision import Decision  # noqa: F401
    return Decision


EXPECT = '`pydantic_graph.beta.decision` is deprecated'

if __name__ == '__main__':
    trigger()
