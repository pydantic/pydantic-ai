"""Pydantic Graph V2.

Ideas:
- Probably need something analogous to Command ...
- Graphs need a way to specify whether to end eagerly or after all forked tasks complete finished
    - In the non-eager case, graph needs a way to specify a reducer for multiple entries to g.end()
        - Default is ignore and warn after the first, but a reducer _can_ be used
    - I think the general case should be a JoinNode[StateT, GraphOutputT, GraphOutputT, Any].

Need to be able to:
* Decision (deterministically decide which node to transition to based on the input, possibly the input type)
* Unpack-fork (send each item of an input sequence to the same node by creating multiple GraphWalkers)
* Broadcast-fork (send the same input to multiple nodes by creating multiple GraphWalkers)
* Join (wait for all upstream GraphWalkers to finish before continuing, reducing their inputs as received)
* Streaming (by providing a channel to deps)
* Interruption
    * Implementation 1: if persistence is necessary, return an Interrupt, and use the `resume` API to continue. Note that you need to snapshot graph state (including all GraphWalkers) to resume
    * Implementation 2: if persistence is not necessary and the implementation can just wait, use channels
* Iteration API (?)
* Command (?)
* Persistence (???) — how should this work with multiple GraphWalkers?
"""
