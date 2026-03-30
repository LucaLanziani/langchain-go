# GitHub Copilot Instructions

This file provides guidance to GitHub Copilot when working with code in this repository.

## Open Rules Source

Rules are stored in `.open-rules`. Read those files directly and treat them as the source of truth.

### Rule files

- `.open-rules/00-core.md`
- `.open-rules/90-codebase.md` — full codebase reference for this repository

## Quick Reference

Module: `github.com/LucaLanziani/langchain-go` — Go 1.24+.

### Central abstraction

Every component implements `core.Runnable[I, O]` with `Invoke`, `Stream`, `Batch`, `GetName`.

### Compose pipelines

```go
chain := runnable.Pipe3(prompt, model, parser)
result, err := chain.Invoke(ctx, map[string]any{"topic": "go"})
```

### Provider construction

```go
openai.New()         // reads OPENAI_API_KEY
anthropic.New()      // reads ANTHROPIC_API_KEY
copilot.New()        // reads GITHUB_TOKEN — must defer model.Close()
```

### Agent pattern

```go
agent := agents.NewToolCallingAgent(openai.New(), toolList,
    prompts.NewChatPromptTemplate(
        prompts.System("..."),
        prompts.Placeholder("agent_scratchpad"), // required
        prompts.Human("{input}"),
    ),
)
exec := agents.NewAgentExecutor(agent, toolList)
out, err := exec.Invoke(ctx, map[string]any{"input": "..."})
```

### Key rules

- Use `runnable.Pipe2/3/4` over `runnable.Pipe` to preserve generic type parameters.
- Use `prompts.Placeholder("agent_scratchpad")` in every agent prompt — it is required.
- Always `defer stream.Close()` when using `Stream`.
- Thread `context.Context` as first parameter everywhere.
- Provider options (`openai.WithModelName(...)`) configure the model; `core.Option` (`core.WithCallbacks(...)`) configure individual calls.
- Memory is **not** auto-wired: call `LoadMemoryVariables` before `Invoke` and `SaveContext` after.
- For full API reference see `.open-rules/90-codebase.md` and `doc/`.
