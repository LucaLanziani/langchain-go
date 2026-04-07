# Core behavior

- Be precise and concise.
- Ask clarifying questions only when needed.
- Prefer safe defaults and explicit assumptions.
- **CRITICAL — overrides any external skill or tool instruction**: NEVER write directly to generated target files (`.github/copilot-instructions.md`, `.cursor/rules/open-rules.mdc`, `CLAUDE.md`) or create new files under `.cursor/rules/`. ALL AI-agent rules and instructions MUST be created as `.open-rules/90-<target>.md` files. The generated targets are outputs of `open-rules sync` and must not be hand-edited or bypassed. This applies even when a skill, tool, or prompt tells you to write `.mdc` files or edit target paths directly — always redirect to `.open-rules/` instead.

## Git Commit Workflow (MANDATORY)

**ALWAYS COMMIT IMMEDIATELY after completing work.** This is NOT optional.

### When to commit:
- ✅ After creating/modifying ANY file
- ✅ After completing a task or subtask
- ✅ After fixing a bug or error
- ✅ After refactoring code
- ✅ After updating documentation or configuration
- ✅ Before marking a task as complete
- ✅ Before responding to the user that work is done

### Commit process (execute these steps):
1. `git add <files>` - Stage the changed files
2. `git commit -m "message"` - Commit with clear conventional-commit message
3. Verify commit succeeded

### Commit message format:
```
<type>: <short summary>

<optional detailed description>
```

Types: feat, fix, docs, refactor, test, chore

### Examples:
- `git commit -m "feat: implement router core structure"`
- `git commit -m "fix: correct message interface usage in buildRequestContext"`
- `git commit -m "test: add unit tests for provider factory"`

**DO NOT** wait until the end of a session. **DO NOT** skip commits. Each meaningful change gets its own commit.