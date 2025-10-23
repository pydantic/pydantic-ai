# Pydantic AI Update Summary - 2025-09-02

## Version Progress
- Your last sync: commit `f7947f4` (Dec 2024)
- Latest upstream: commit `66fa21ab` (Jan 2025) (user says: "press X for doubt")
- New releases: v0.5.1 through v1.0.0b1 (now in beta!)

## Key Changes Summary (187 commits)

### 🔥 Breaking Changes
- **Python 3.9 dropped** - Now requires Python 3.10+
- Many dataclasses now keyword-only
- `OpenAIModel` deprecated → use `OpenAIChatModel`
- `GeminiModel` deprecated → use `GoogleModel`
- `result_type` parameter removed from `Agent`
- `data` removed from `FinalResult`
- `format_as_xml` module removed
- `Usage` deprecated → use `RequestUsage`/`RunUsage`
- Model names now require provider prefix (e.g., `openai:gpt-4`)

### ✨ Major Features
1. **Temporal Integration** - Agent can now run in Temporal workflows
2. **Human-in-the-loop (HITL)** - Tool call approval support
3. **MCP Improvements** - Better server handling, elicitation callbacks
4. **Logfire Integration** - Now included with pydantic-ai package
5. **Web Search Tool** - Native Google URL context support
6. **Cerebras Provider** - New model provider added
7. **MockOpenAIResponses** - For testing purposes
8. **Newsletter Form** - Added to docs

### 🐛 Notable Fixes
- Gemini streaming with code execution fixed
- Google service account auth improvements
- Retry behavior improved (proper response closing)
- Video token counting in Google models
- MCP server exit handling
- Anthropic thinking parts with Bedrock

### 📚 Documentation
- Reorganized nav hierarchy for better UX
- Added AGENTS.md file
- Beta release warnings added
- Temporal docs updated
- HITL examples improved

### 🔧 Internal Improvements
- OpenTelemetry GenAI conventions updated
- Test suite speed improvements
- Coverage improvements
- CI matrix improvements
- Instrumentation v2 as default

## Update Procedure

To safely update while preserving your custom files:

```bash
# 1. Stash your custom files
git stash push -m "Custom files before update" CLAUDE.md

# 2. Merge upstream changes
git merge origin/main --no-commit

# 3. Handle CLAUDE.md conflict
# Get Pydantic AI's latest CLAUDE.md as CLAUDE-original.md
git checkout origin/main -- CLAUDE.md
mv CLAUDE.md CLAUDE-original.md

# 4. Restore your CLAUDE.md
git stash pop
# This will restore your CLAUDE.md

# 5. Add both files and commit
git add CLAUDE.md CLAUDE-original.md
git commit -m "Update to Pydantic AI v1.0.0b1 while preserving custom files"
```

## Files to Watch
Your custom directories are safe:
- `claude/` - Your Claude Code workspace
- `prometheus/` - Your design docs
- `CLAUDE.md` - Your instructions (preserved)
- `CLAUDE-original.md` - Pydantic AI's instructions (updated)