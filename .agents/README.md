# `.agents/` Directory

This directory contains cross-agent project configuration that works with any coding agent (Claude Code, OpenCode, Codex, etc).

## Skills

`.agents/skills/` is the canonical location for project skills. `.claude/skills` is a symlink pointing here for Claude Code compatibility.

### Merge conflict resolution

If you get a `CONFLICT (file/directory)` on `.claude/skills` after a merge, move any new skills to `.agents/skills/` and restore the symlink:

```bash
rm -rf .claude/skills
ln -s ../.agents/skills .claude/skills
```
