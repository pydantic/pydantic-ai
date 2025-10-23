# CLAUDE.md

## Purpose

This fork of Pydantic AI is a **workspace for analyzing and understanding the Pydantic AI codebase and documentation**. Both Claude Code Mobile (iOS) and Claude Code Desktop (Mac) collaborate on this analysis, creating reports and documentation in the `claude/` directory.

**What we do here:**
- Explore and analyze the Pydantic AI codebase
- Document how various components work
- Create guides and explainers for complex features
- Share findings between Mobile and Desktop instances

**What we don't do:**
- Develop or modify Pydantic AI itself (this is a read-only analysis fork)
- Run tests or CI/CD workflows (removed for simplicity)

## Repository Setup

- **Fork**: `ericksonc/pydantic-ai`
- **Upstream**: `pydantic/pydantic-ai` (official repo, for reference only)
- **Working Directory**: `claude/` - all reports and documentation go here
- **No CI/CD**: GitHub Actions workflows have been removed (we only sync markdown files)

## Sync Workflow

### For Claude Code Mobile (iOS)

When you create or update reports:

1. **Edit files** in the `claude/` directory
2. **Commit changes** with a descriptive message
3. **Push directly to main**:
   ```bash
   git add claude/
   git commit -m "Add/update report: [description]"
   git push origin main
   ```

**Important**: Work directly on `main` branch. No need to create feature branches for simple markdown sync.

### For Claude Code Desktop (Mac)

To sync reports from Mobile:

**Option 1: Use the sync script (recommended)**
```bash
./sync-mobile-reports.sh
```

**Option 2: Manual pull**
```bash
git pull origin main
```

The sync script will:
- Fetch latest changes from the fork
- Check for updates in `claude/` directory
- Pull changes automatically if found
- Display sync status

## File Organization

```
claude/
├── [various report files].md       # Reports created by either Mobile or Desktop
├── modelmessage/                   # Organized documentation subdirectories
│   └── [related docs].md
└── user/                           # User-specific documentation
    └── [user docs].md
```

## How Analysis & Sync Works

**Analysis Flow:**
- Both Mobile and Desktop Claude analyze the Pydantic AI codebase
- Reports and findings are written to `claude/` directory
- Each instance can build on the other's work

**Sync Flow:**
```
Mobile: Analyzes code → Creates report → Commits & pushes to main
                                            ↓
                            GitHub Fork (ericksonc/pydantic-ai)
                                            ↓
Desktop: Runs ./sync-mobile-reports.sh → Pulls reports → Can continue analysis
```

## Important Notes

- **Both instances work on `main`**: No branching needed for markdown sync
- **Conflicts are rare**: Mobile and Desktop typically work on different files
- **If conflicts occur**: Desktop should pull first, resolve manually, then push
- **The sync script is idempotent**: Safe to run multiple times
- **Original CLAUDE.md**: See `CLAUDE-original.md` for Pydantic AI development instructions

## Reference Links

- Fork: https://github.com/ericksonc/pydantic-ai
- Upstream: https://github.com/pydantic/pydantic-ai
- Claude Code: https://claude.com/claude-code
