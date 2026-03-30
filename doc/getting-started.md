# Getting Started

This guide walks you through installing langchain-go and building your first AI application in Go.

---

## Prerequisites

- Go 1.24 or later
- An API key for at least one provider (OpenAI, Anthropic, or GitHub Copilot)

---

## Installation

```bash
go get github.com/LucaLanziani/langchain-go
```

---

## Environment Setup

Export your API key before running any example:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# GitHub Copilot
export GITHUB_TOKEN="ghp_..."

# LangSmith tracing (optional)
export LANGCHAIN_API_KEY="ls__..."
export LANGCHAIN_PROJECT="my-project"
```

---

## Example 1: Simple chat

The quickest way to talk to a model:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/LucaLanziani/langchain-go/core"
    "github.com/LucaLanziani/langchain-go/providers/openai"
)

func main() {
    ctx   := context.Background()
    model := openai.New()

    resp, err := model.Invoke(ctx, []core.Message{
        core.NewSystemMessage("You are a concise assistant."),
        core.NewHumanMessage("What is Go?"),
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(resp.Content)
}
```

Run it:

```bash
go run main.go
```

---

## Example 2: Prompt template + model + parser

Compose a reusable pipeline using the `Pipe3` combinator:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/LucaLanziani/langchain-go/outputparsers"
    "github.com/LucaLanziani/langchain-go/prompts"
    "github.com/LucaLanziani/langchain-go/providers/openai"
    "github.com/LucaLanziani/langchain-go/runnable"
)

func main() {
    ctx := context.Background()

    prompt := prompts.NewChatPromptTemplate(
        prompts.System("You are a helpful assistant that tells jokes."),
        prompts.Human("Tell me a short joke about {topic}"),
    )
    model  := openai.New()
    parser := outputparsers.NewStringOutputParser()

    chain := runnable.Pipe3(prompt, model, parser)

    result, err := chain.Invoke(ctx, map[string]any{"topic": "golang"})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result)
}
```

---

## Example 3: Streaming

Receive output token by token:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/LucaLanziani/langchain-go/core"
    "github.com/LucaLanziani/langchain-go/providers/openai"
)

func main() {
    ctx   := context.Background()
    model := openai.New()

    stream, err := model.Stream(ctx, []core.Message{
        core.NewSystemMessage("You are a helpful assistant."),
        core.NewHumanMessage("Write a haiku about Go programming."),
    })
    if err != nil {
        log.Fatal(err)
    }
    defer stream.Close()

    for {
        chunk, ok, err := stream.Next()
        if err != nil { log.Fatal(err) }
        if !ok { break }
        fmt.Print(chunk.Content)
    }
    fmt.Println()
}
```

---

## Example 4: Agent with tools

Give the model a calculator tool and let it decide when to use it:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/LucaLanziani/langchain-go/agents"
    "github.com/LucaLanziani/langchain-go/prompts"
    "github.com/LucaLanziani/langchain-go/providers/openai"
    "github.com/LucaLanziani/langchain-go/tools"
)

func main() {
    ctx := context.Background()

    calc := tools.NewTool(
        "calculator",
        "Evaluate a mathematical expression. Input: the expression as a string.",
        func(_ context.Context, input string) (string, error) {
            // In a real app, parse and evaluate the expression.
            return "42", nil
        },
    )

    prompt := prompts.NewChatPromptTemplate(
        prompts.System("You are a math assistant. Use the calculator tool for arithmetic."),
        prompts.Placeholder("agent_scratchpad"),
        prompts.Human("{input}"),
    )

    agent := agents.NewToolCallingAgent(openai.New(), []tools.Tool{calc}, prompt)
    exec  := agents.NewAgentExecutor(agent, []tools.Tool{calc},
        agents.WithMaxIterations(10),
    )

    result, err := exec.Invoke(ctx, map[string]any{"input": "What is 6 times 7?"})
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(result["output"])
}
```

---

## Example 5: RAG — question answering over documents

Build a pipeline that retrieves relevant context before answering:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/LucaLanziani/langchain-go/chains"
    "github.com/LucaLanziani/langchain-go/core"
    "github.com/LucaLanziani/langchain-go/prompts"
    "github.com/LucaLanziani/langchain-go/providers/openai"
    "github.com/LucaLanziani/langchain-go/retrievers"
    "github.com/LucaLanziani/langchain-go/textsplitters"
    "github.com/LucaLanziani/langchain-go/vectorstores/inmemory"
)

func main() {
    ctx := context.Background()

    // 1. Documents
    docs := []*core.Document{
        core.NewDocument("Go was created at Google in 2009 by Robert Griesemer, Rob Pike, and Ken Thompson."),
        core.NewDocument("Go 1.18 introduced generics (type parameters) in March 2022."),
        core.NewDocument("Go has built-in concurrency via goroutines and channels."),
    }

    // 2. Chunk + embed + store
    splitter := textsplitters.NewRecursiveCharacterTextSplitter(300, 30)
    chunks   := splitter.SplitDocuments(docs)
    store    := inmemory.New(openai.NewEmbeddings())
    if _, err := store.AddDocuments(ctx, chunks); err != nil {
        log.Fatal(err)
    }

    // 3. Wire up the QA chain
    retriever := retrievers.NewVectorStoreRetriever(store, 3)
    qaPrompt  := prompts.NewChatPromptTemplate(
        prompts.System("Answer the question using ONLY this context:\n\n{context}"),
        prompts.Human("{query}"),
    )
    qaChain := chains.NewRetrievalQA(
        retriever,
        chains.NewLLMChain(openai.New(), qaPrompt),
    )

    // 4. Ask
    answer, err := qaChain.Invoke(ctx, map[string]any{
        "query": "Who created Go and when?",
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(answer)
}
```

---

## Next Steps

| Topic | Guide |
|---|---|
| System architecture & diagrams | [Architecture](architecture.md) |
| All core interfaces explained | [Core Concepts](core-concepts.md) |
| OpenAI, Anthropic, GitHub Copilot | [Providers](providers.md) |
| Building pipelines | [Chains & Runnables](chains-and-runnables.md) |
| ReAct and ToolCalling agents | [Agents](agents.md) |
| Conversation history | [Memory](memory.md) |
| Creating tools | [Tools](tools.md) |
| Embeddings, vector stores, retrieval | [RAG Pipeline](rag.md) |
| Observability and LangSmith | [Callbacks & Tracing](callbacks-and-tracing.md) |
