"""Saved workflows: the file format, the per-run library, and the `<available_workflows>` block.

A saved workflow is a `<name>.py` file whose top level assigns a `meta` dict literal, followed by
the script. `meta` is read with `ast.literal_eval`, never executed, so loading a library runs no
code. The directory is model-writable, so an invalid file is recorded and skipped, never fatal.
"""

from __future__ import annotations

import ast
import json
import posixpath
import pprint
import re
import warnings
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ConfigDict, JsonValue, TypeAdapter, ValidationError, with_config
from typing_extensions import NotRequired, TypedDict

from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness._workspace import workspace_path

_NAME_PATTERN = re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')
_PARSE_FLAGS = ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT


@with_config(ConfigDict(extra='forbid'))
class _WorkflowMeta(TypedDict):
    """The `meta` dict at the top of a saved workflow file."""

    name: str
    description: str
    when_to_use: NotRequired[str]
    args: NotRequired[dict[str, JsonValue]]
    agents: NotRequired[list[str]]
    returns: NotRequired[str]


_META_ADAPTER = TypeAdapter(_WorkflowMeta)


@dataclass(frozen=True, kw_only=True)
class SavedWorkflow:
    """One saved workflow: its `meta` fields and the script source they head."""

    name: str
    """The workflow's name, equal to its file stem: lowercase letters, digits, `-` and `_`."""

    description: str
    """What the workflow does, shown to the model in `<available_workflows>`."""

    source: str
    """The whole file, `meta` included; this is what runs."""

    when_to_use: str | None = None
    """When the model should reach for it."""

    args: dict[str, JsonValue] | None = None
    """JSON schema of the `args` the workflow expects. Only its top-level `required` keys are checked."""

    agents: tuple[str, ...] = ()
    """Catalog sub-agents the script calls; a library drops a workflow naming one the catalog lacks."""

    returns: str | None = None
    """What the workflow's final expression evaluates to."""

    path: str | None = None
    """Where the file was loaded from, when it was."""

    @classmethod
    def parse(cls, source: str, *, path: str | None = None) -> SavedWorkflow:
        """Parse a saved workflow file, raising `ValueError` naming what is wrong with it."""
        try:
            tree = compile(source, path or '<workflow>', 'exec', _PARSE_FLAGS)
        except SyntaxError as error:
            raise ValueError(f'syntax error on line {error.lineno}: {error.msg}') from error
        assert isinstance(tree, ast.Module)
        meta_nodes = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(target := node.targets[0], ast.Name)
            and target.id == 'meta'
        ]
        if len(meta_nodes) != 1:
            raise ValueError('expected exactly one top-level `meta = {...}` assignment')
        try:
            raw: object = ast.literal_eval(meta_nodes[0])
        except ValueError as error:
            raise ValueError('`meta` must be a dict literal') from error
        try:
            meta = _META_ADAPTER.validate_python(raw)
        except ValidationError as error:
            raise ValueError(f'invalid `meta`: {_validation_summary(error)}') from error
        name = meta['name']
        if not _NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f'invalid name {name!r}: use 1-64 lowercase letters, digits, `-` and `_`, starting with a letter or digit'
            )
        return cls(
            name=name,
            description=meta['description'],
            source=source,
            when_to_use=meta.get('when_to_use'),
            args=meta.get('args'),
            agents=tuple(meta.get('agents', ())),
            returns=meta.get('returns'),
            path=path,
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        description: str,
        code: str,
        when_to_use: str | None = None,
        args: dict[str, JsonValue] | None = None,
        agents: Sequence[str] = (),
        returns: str | None = None,
    ) -> SavedWorkflow:
        """Build a workflow from its fields, rendering the `meta` header above `code`.

        Round-trips through `parse`, so whatever this returns loads back unchanged.
        """
        meta: dict[str, object] = {'name': name, 'description': description}
        optional: dict[str, object | None] = {
            'when_to_use': when_to_use,
            'args': args,
            'agents': list(agents) or None,
            'returns': returns,
        }
        meta.update({key: value for key, value in optional.items() if value is not None})
        header = f'meta = {pprint.pformat(meta, sort_dicts=False, width=100)}'
        return cls.parse(f'{header}\n\n{code.strip()}\n')

    def missing_args(self, args: dict[str, JsonValue]) -> list[str]:
        """The `required` keys of the `args` schema absent from `args`."""
        required = (self.args or {}).get('required')
        if not isinstance(required, list):
            return []
        return [key for key in required if isinstance(key, str) and key not in args]


def _validation_summary(error: ValidationError) -> str:
    return '; '.join(
        f'{".".join(str(part) for part in item["loc"]) or "meta"}: {item["msg"]}' for item in error.errors()
    )


@dataclass
class WorkflowLibrary:
    """The saved workflows available to one run, and why any file was left out."""

    workflows: dict[str, SavedWorkflow] = field(default_factory=dict[str, SavedWorkflow])
    """Valid workflows by name, in load order (directory, then file name), so the listing is stable across runs."""

    errors: dict[str, str] = field(default_factory=dict[str, str])
    """Why each skipped file was skipped, by path."""

    @classmethod
    async def load(
        cls, workspace: Workspace, directories: Sequence[str | Path], *, agent_names: Collection[str]
    ) -> WorkflowLibrary:
        """Read every `*.py` directly inside `directories`, in the workspace.

        A directory that does not exist contributes nothing. A file that fails to parse, whose
        `name` is not its stem, that repeats an earlier name, or that names a sub-agent missing
        from `agent_names` is recorded in `errors` with a warning, and the rest still load.
        """
        library = cls()
        for configured in directories:
            directory = await workspace.resolve(workspace_path(Path(configured)))
            try:
                entries = await workspace.list_dir(directory)
            except (FileNotFoundError, NotADirectoryError):
                continue
            for entry in sorted(entries, key=lambda entry: entry.name):
                stem, suffix = posixpath.splitext(entry.name)
                if entry.is_dir or suffix != '.py':
                    continue
                try:
                    workflow = SavedWorkflow.parse(await workspace.read_text(entry.path), path=entry.path)
                    if workflow.name != stem:
                        raise ValueError(f'`meta["name"]` is {workflow.name!r} but the file is named {stem!r}')
                    if workflow.name in library.workflows:
                        raise ValueError(f'repeats the name of {library.workflows[workflow.name].path}')
                    if missing := [agent for agent in workflow.agents if agent not in agent_names]:
                        raise ValueError(f'names sub-agents missing from the catalog: {", ".join(missing)}')
                except (ValueError, UnicodeDecodeError) as error:
                    library.errors[entry.path] = str(error)
                    warnings.warn(f'Skipping saved workflow {entry.path}: {error}', UserWarning, stacklevel=2)
                    continue
                library.workflows[workflow.name] = workflow
        return library

    def render(self) -> str | None:
        """The `<available_workflows>` instructions block, or `None` when there is nothing to list."""
        if not self.workflows:
            return None
        blocks: list[str] = []
        for workflow in self.workflows.values():
            lines = [f'name: {workflow.name}', f'description: {workflow.description}']
            if workflow.when_to_use:
                lines.append(f'when_to_use: {workflow.when_to_use}')
            if workflow.args is not None:
                lines.append(f'args: {json.dumps(workflow.args, sort_keys=True)}')
            if workflow.returns:
                lines.append(f'returns: {workflow.returns}')
            blocks.append('<workflow>\n' + '\n'.join(lines) + '\n</workflow>')
        intro = (
            'Saved workflows: run one with the `run_workflow` tool by `name` (plus `args`), '
            'or from inside a script with `await workflow(name, args)`.'
        )
        return '<available_workflows>\n' + intro + '\n' + '\n'.join(blocks) + '\n</available_workflows>'
