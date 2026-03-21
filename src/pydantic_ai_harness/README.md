TODOs and plan for this repo

## intro

TODO

## glossary

AICA: AI Code Assistant
Research: AICAs are able to do research via WebFetching, WebSearching or using the `gh` CLI to look up issues, PRs and comments on github, as well as explore code they don't locally have access to.

## mechanical 

- [ ] start a new python package (`pydantic-harness`)
    - [ ] `uv init`


## automated code factory

- the optimal flow is issue gets opened and discussed -> when ready, AICA is triggered to open a PR to close the issue with only one file (`PLAN.md`)
- `PLAN.md` is reviewed (inline comments, PR discussion) -> AICA is triggered to update the plan
- when the plan is ready, AICA is triggered to implement the plan
- AICA must loop to review the changes are according to the plan, identify any new questions
    - more about looping under [ralph loop](#ralph-loop) below.

### how does the AICA get triggered? via the label system

We use labels to trigger AICA actions, at the end of each action AICA must remove the label that triggered it.

Labels:
- when an issue is labeled with `aica:research`, AICA is triggered to answer questions posed in the issue (body and/or comments) by researching and linking sources, emphasis on **linking sources** for every claim it makes
- when an issue is labeled with `aica:write-plan`, AICA is triggered to create a `PLAN.md` in the issue's branch
- when a PR is labeled with `aica:update-plan`, AICA is triggered to update the `PLAN.md` in the PR's branch according to the latest discussions in the PR
    - note that AICA must be able to do research in this step as well before updating the plan, if any questions require it
    - any findings and decisions made must be documented in a new PR comment, not as part of the thread
    - the reason for this is so we can resolve threads. if we document decisions in threads, we have to keep the threads open, which bloats the PR page and creates noise for future fetches of the PR. 
- when a PR is labeled with `aica:implement-plan`, AICA is triggered to implement the plan in `PLAN.md` in the PR's branch

### ralph loop

TODO