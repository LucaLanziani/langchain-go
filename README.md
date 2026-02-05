# langchain-go

**LangChain for Go** -- build production-grade AI agents as single, high-performance binaries.

langchain-go brings the battle-tested LangChain framework natively to Go, giving you
agents, chains, tools, memory, vector stores, and LLM integrations without leaving the
Go ecosystem.

## Quickstart

```bash
go get github.com/langchain-go/langchain-go
```

### Simple chain (prompt + model + parser)

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/langchain-go/langchain-go/outputparsers"
    "github.com/langchain-go/langchain-go/prompts"
    "github.com/langchain-go/langchain-go/providers/openai"
    "github.com/langchain-go/langchain-go/runnable"
)

func main() {
    ctx := context.Background()

    prompt := prompts.NewChatPromptTemplate(
        prompts.System("You are a helpful assistant that tells jokes."),
        prompts.Human("Tell me a short joke about {topic}"),
    )
    model  := openai.New()
    parser := outputparsers.NewStringOutputParser()

    // Compose: prompt -> model -> parser
    chain := runnable.Pipe3(prompt, model, parser)

    result, err := chain.Invoke(ctx, map[string]any{"topic": "golang"})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result)
}
```

### Agent with tool use

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/langchain-go/langchain-go/agents"
    "github.com/langchain-go/langchain-go/prompts"
    "github.com/langchain-go/langchain-go/providers/openai"
    "github.com/langchain-go/langchain-go/tools"
)

func main() {
    ctx := context.Background()

    calc := tools.NewTool("calculator", "Evaluate math expressions.",
        func(_ context.Context, input string) (string, error) {
            return "42", nil // replace with real eval
        },
    )

    prompt := prompts.NewChatPromptTemplate(
        prompts.System("You are a helpful assistant. Use tools when needed."),
        prompts.Placeholder("agent_scratchpad"),
        prompts.Human("{input}"),
    )

    agent   := agents.NewToolCallingAgent(openai.New(), []tools.Tool{calc}, prompt)
    exec    := agents.NewAgentExecutor(agent, []tools.Tool{calc})

    result, err := exec.Invoke(ctx, map[string]any{"input": "What is 6 * 7?"})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result["output"])
}
```

## Architecture

langchain-go follows Go-idiomatic design principles:

| Python LangChain | langchain-go | Notes |
|---|---|---|
| `invoke` / `ainvoke` | `Invoke` | Single method, use goroutines for concurrency |
| `stream` / `astream` | `Stream` | Returns `*StreamIterator[T]` |
| `batch` / `abatch` | `Batch` | Parallel by default with `MaxConcurrency` control |
| `\|` operator (LCEL) | `runnable.Pipe2`, `Pipe3`, `Pipe4` | Type-safe composition |
| `RunnableParallel` | `runnable.NewParallel` | Fan-out / fan-in |
| `RunnableLambda` | `runnable.NewLambda` | Wrap any Go function |
| `RunnableBranch` | `runnable.NewBranch` | Conditional routing |
| `**kwargs` | Functional options (`...core.Option`) | `WithTemperature(0.7)` |
| `async/await` | `context.Context` | Cancellation and timeouts |

### Core interface

Every component implements `Runnable[I, O]`:

```go
type Runnable[I, O any] interface {
    Invoke(ctx context.Context, input I, opts ...Option) (O, error)
    Stream(ctx context.Context, input I, opts ...Option) (*StreamIterator[O], error)
    Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error)
    GetName() string
}
```

## Packages

| Package | Description |
|---|---|
| `core` | Core types: messages, documents, Runnable interface, config, callbacks |
| `prompts` | Prompt templates (`PromptTemplate`, `ChatPromptTemplate`, `MessagesPlaceholder`) |
| `outputparsers` | Output parsers (`StringOutputParser`, `JSONOutputParser`) |
| `runnable` | Composition primitives (Sequence, Parallel, Lambda, Passthrough, Branch) |
| `llms` | Chat model interface and option types |
| `providers/openai` | OpenAI chat models and embeddings |
| `providers/anthropic` | Anthropic/Claude chat models |
| `tools` | Tool interface and typed tool factory |
| `agents` | Agent implementations (ToolCalling, ReAct) and AgentExecutor |
| `chains` | Common chain patterns (LLMChain, StuffDocuments, RetrievalQA) |
| `memory` | Conversation memory (Buffer, Window) |
| `embeddings` | Embedder interface |
| `vectorstores` | Vector store interface + in-memory implementation |
| `retrievers` | Retriever interface wrapping vector stores |
| `textsplitters` | Text splitting utilities |
| `callbacks` | Callback handlers (Stdout, LangSmith) |

## Providers

### OpenAI

```go
import "github.com/langchain-go/langchain-go/providers/openai"

model := openai.New(
    openai.WithAPIKey("sk-..."),       // or OPENAI_API_KEY env var
    openai.WithModelName("gpt-4o"),
)
```

### Anthropic

```go
import "github.com/langchain-go/langchain-go/providers/anthropic"

model := anthropic.New(
    anthropic.WithAPIKey("sk-..."),    // or ANTHROPIC_API_KEY env var
    anthropic.WithModelName("claude-sonnet-4-20250514"),
    anthropic.WithMaxTokens(4096),
)
```

## Streaming

```go
stream, _ := model.Stream(ctx, messages)
for {
    chunk, ok, err := stream.Next()
    if err != nil { log.Fatal(err) }
    if !ok { break }
    fmt.Print(chunk.Content)
}
```

## Tools

Create tools from Go functions with automatic JSON Schema generation:

```go
type SearchArgs struct {
    Query string `json:"query" description:"The search query"`
    Limit int    `json:"limit,omitempty" description:"Max results"`
}

search := tools.NewTypedTool("search", "Search the web", SearchArgs{},
    func(ctx context.Context, args SearchArgs) (string, error) {
        return fmt.Sprintf("Results for: %s", args.Query), nil
    },
)
```

## Memory

```go
mem := memory.NewConversationBufferMemory()
mem.SaveContext(ctx,
    map[string]any{"input": "Hello"},
    map[string]any{"output": "Hi there!"},
)
vars, _ := mem.LoadMemoryVariables(ctx, nil)
fmt.Println(vars["history"]) // "Human: Hello\nAI: Hi there!"
```

## Callbacks and Observability

```go
// Stdout debugging
result, err := chain.Invoke(ctx, input,
    core.WithCallbacks(callbacks.NewStdoutHandler()),
)

// LangSmith tracing
result, err := chain.Invoke(ctx, input,
    core.WithCallbacks(callbacks.NewLangSmithHandler("my-project")),
)
```

## Examples

See the [`examples/`](examples/) directory:

- **[simple_chain](examples/simple_chain/)** -- Prompt + model + parser chain
- **[agent_tool_use](examples/agent_tool_use/)** -- Agent with tools and callbacks
- **[streaming](examples/streaming/)** -- Streaming chat completion
- **[rag_pipeline](examples/rag_pipeline/)** -- Full RAG with vector store

## Requirements

- Go 1.22 or later
- API keys for the providers you want to use (OpenAI, Anthropic)

## License

MIT
