# langchain-go Documentation

**LangChain for Go** — build production-grade AI agents as single, high-performance binaries.

langchain-go brings the battle-tested LangChain framework natively to Go, giving you agents, chains, tools, memory, vector stores, and LLM integrations without leaving the Go ecosystem.

---

## Documentation Index

| Document | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation, quickstart examples, and first steps |
| [Architecture](architecture.md) | System design, component overview, and data-flow diagrams |
| [Core Concepts](core-concepts.md) | `Runnable`, `Message`, `RunnableConfig`, callbacks — the foundational interfaces |
| [Providers](providers.md) | OpenAI, Anthropic, and GitHub Copilot LLM integrations |
| [Chains & Runnables](chains-and-runnables.md) | Composing pipelines with `Sequence`, `Branch`, `Parallel`, and predefined chains |
| [Agents](agents.md) | ReAct and ToolCalling agents, `AgentExecutor` |
| [Memory](memory.md) | Conversation memory: buffer, window, and chat history |
| [Tools](tools.md) | Creating simple and typed tools for agents |
| [RAG Pipeline](rag.md) | Embeddings, vector stores, retrievers, and text splitters |
| [Callbacks & Tracing](callbacks-and-tracing.md) | Observability, callbacks, and LangSmith integration |

---

## Package Overview

```
github.com/LucaLanziani/langchain-go
├── core/           — Fundamental interfaces: Runnable, Message, Config, Callbacks
├── llms/           — ChatModel interface and result types
├── providers/
│   ├── openai/     — OpenAI GPT models + embeddings
│   ├── anthropic/  — Anthropic Claude models
│   └── github-copilot/ — GitHub Copilot models
├── prompts/        — Chat and text prompt templates
├── chains/         — Predefined chains: LLMChain, StuffDocuments, RetrievalQA
├── runnable/       — LCEL-style composition: Sequence, Branch, Parallel, Lambda
├── agents/         — ReAct agent, ToolCallingAgent, AgentExecutor
├── tools/          — Tool interface and structured tool builders
├── memory/         — ConversationBufferMemory, ConversationWindowMemory
├── embeddings/     — Embedder interface
├── vectorstores/   — VectorStore interface + in-memory implementation
├── retrievers/     — VectorStoreRetriever
├── textsplitters/  — RecursiveCharacterTextSplitter
├── outputparsers/  — String and JSON output parsers
├── callbacks/      — Callback manager and LangSmith handler
└── internal/       — Internal utilities
```

---

## Design Philosophy

langchain-go follows Go-idiomatic design principles:

- **Single interface, no async split** — Go's goroutines replace Python's `async/await`. Every method is synchronous; use goroutines and `context.Context` for concurrency.
- **Generics for type safety** — `Runnable[I, O]` is generic, removing the `any`-typed boilerplate prevalent in many Go AI libraries.
- **Zero framework magic** — No reflection-based dependency injection, no global registries. Components are plain Go structs you compose explicitly.
- **Standard library first** — Minimal external dependencies; mostly uses the standard library plus a small number of well-known packages.
