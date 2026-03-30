# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Open Rules Source

Rules are stored in `.open-rules`. Read those files directly and treat them as the source of truth. Do not rely on copied content in this file.

### Rule files

- `.open-rules/00-core.md`
- `.open-rules/90-codebase.md` — full codebase reference: all packages, types, constructors, patterns, and conventions

## Commands

```bash
# Run all tests
go test ./...

# Run tests for a specific package
go test ./runnable/...

# Run a single test
go test ./runnable/ -run TestSequence

# Run tests with verbose output
go test -v ./...

# Build
go build ./...

# Lint (if golangci-lint is installed)
golangci-lint run
```

## Architecture

Module: `github.com/LucaLanziani/langchain-go` — requires Go 1.24+.

### Core abstraction

Every component implements `core.Runnable[I, O]` (`core/runnable.go`), providing `Invoke`, `Stream`, `Batch`, and `GetName`. This is the single interface that wires the whole system together.

### Composition layer (`runnable/`)

`runnable.Pipe2/Pipe3/Pipe4` — type-safe LCEL-style chaining. Additional primitives: `Parallel` (fan-out/fan-in), `Lambda` (wrap any function), `Passthrough`, `Branch` (conditional routing). These compose any `Runnable` implementations.

### LLM layer (`llms/`, `providers/`)

`llms.ChatModel` extends `Runnable[[]core.Message, core.AIMessage]`. Provider implementations live under `providers/openai`, `providers/anthropic`, and `providers/github-copilot`. Options use the functional options pattern (`...core.Option`).

### Prompt / Parser layer

`prompts.ChatPromptTemplate` (`prompts/chat.go`) — implements `Runnable[map[string]any, []core.Message]`. Output parsers (`outputparsers/`) implement `Runnable[core.AIMessage, T]`.

### Agents (`agents/`)

`ToolCallingAgent` and `ReActAgent` implement the agent loop. `AgentExecutor` wraps an agent and a tool list, handling the tool-call/observation cycle.

### Memory, Vector stores, Retrievers

`memory/` — `ConversationBufferMemory`, `ConversationWindowMemory`. `vectorstores/inmemory` — reference in-memory implementation of the `VectorStore` interface. `retrievers/` — wraps vector stores as `Runnable[string, []core.Document]`.

### Callbacks (`callbacks/`)

`CallbackManager` is threaded through `core.RunnableConfig` via `core.WithCallbacks(...)`. Built-in handlers: `StdoutHandler`, `LangSmithHandler`.
