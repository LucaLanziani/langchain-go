# Architecture

This document describes the high-level architecture of langchain-go, its component model, and the key data flows.

---

## Component Overview

```mermaid
graph TB
    subgraph "Application Layer"
        App["Your Application"]
    end

    subgraph "Agent Layer"
        Executor["AgentExecutor"]
        ReAct["ReActAgent"]
        ToolCalling["ToolCallingAgent"]
    end

    subgraph "Chain Layer"
        LLMChain["LLMChain"]
        StuffDocs["StuffDocumentsChain"]
        RetrievalQA["RetrievalQA"]
    end

    subgraph "Runnable Combinators"
        Sequence["Sequence (Pipe)"]
        Branch["Branch"]
        Parallel["Parallel"]
        Lambda["Lambda"]
        Passthrough["Passthrough / Assign"]
    end

    subgraph "Core Primitives"
        Runnable["Runnable[I,O]"]
        Messages["Messages"]
        Config["RunnableConfig"]
        Callbacks["CallbackHandler"]
    end

    subgraph "Model Layer"
        ChatModel["ChatModel interface"]
        OpenAI["openai.ChatModel"]
        Anthropic["anthropic.ChatModel"]
        Copilot["github-copilot.ChatModel"]
    end

    subgraph "Prompt Layer"
        ChatPrompt["ChatPromptTemplate"]
        Template["PromptTemplate"]
        Placeholder["MessagesPlaceholder"]
    end

    subgraph "Tools"
        Tool["Tool interface"]
        StructuredTool["StructuredTool"]
        TypedTool["NewTypedTool[T]"]
    end

    subgraph "Memory"
        Memory["Memory interface"]
        Buffer["ConversationBufferMemory"]
        Window["ConversationWindowMemory"]
        ChatHistory["ChatMessageHistory"]
    end

    subgraph "RAG Stack"
        Embedder["Embedder interface"]
        VectorStore["VectorStore interface"]
        InMemory["inmemory.Store"]
        Retriever["VectorStoreRetriever"]
        TextSplitter["RecursiveCharacterTextSplitter"]
    end

    subgraph "Observability"
        Manager["callbacks.Manager"]
        LangSmith["LangSmithHandler"]
        Stdout["StdoutHandler"]
    end

    App --> Executor
    App --> LLMChain
    App --> Sequence
    Executor --> ReAct
    Executor --> ToolCalling
    ReAct --> ChatModel
    ToolCalling --> ChatModel
    LLMChain --> ChatModel
    LLMChain --> ChatPrompt
    RetrievalQA --> Retriever
    RetrievalQA --> LLMChain
    Sequence --> Runnable
    ChatModel --> OpenAI
    ChatModel --> Anthropic
    ChatModel --> Copilot
    ChatPrompt -.-> Messages
    Retriever --> VectorStore
    VectorStore --> InMemory
    InMemory --> Embedder
    Buffer --> ChatHistory
    Window --> ChatHistory
    Manager --> LangSmith
    Manager --> Stdout
    Callbacks -.-> Manager
```

---

## The Runnable Interface — The Backbone

Every component in langchain-go implements a single generic interface:

```mermaid
classDiagram
    class Runnable {
        <<interface>>
        +Invoke(ctx, I, ...Option) (O, error)
        +Stream(ctx, I, ...Option) (*StreamIterator[O], error)
        +Batch(ctx, []I, ...Option) ([]O, error)
        +GetName() string
    }

    class ChatModel {
        <<interface>>
        +Generate(ctx, [][]Message, ...Option) (*ChatResult, error)
        +BindTools(...ToolDefinition) ChatModel
        +WithStructuredOutput(schema) ChatModel
    }

    class ChatPromptTemplate {
        +FormatMessages(inputs) ([]Message, error)
    }

    class AgentExecutor {
        +agent Agent
        +tools []Tool
        +maxIterations int
    }

    class Sequence {
        +steps []step
    }

    Runnable <|-- ChatModel
    Runnable <|-- ChatPromptTemplate
    Runnable <|-- AgentExecutor
    Runnable <|-- Sequence
    ChatModel <|-- OpenAIChatModel
    ChatModel <|-- AnthropicChatModel
    ChatModel <|-- CopilotChatModel
```

---

## Data Flow: Simple Chain

```mermaid
sequenceDiagram
    participant App
    participant Sequence
    participant ChatPromptTemplate
    participant ChatModel
    participant OutputParser

    App->>Sequence: Invoke({"topic": "golang"})
    Sequence->>ChatPromptTemplate: Invoke({"topic": "golang"})
    ChatPromptTemplate-->>Sequence: []Message{System, Human}
    Sequence->>ChatModel: Invoke([]Message)
    ChatModel-->>Sequence: AIMessage{Content: "..."}
    Sequence->>OutputParser: Invoke(AIMessage)
    OutputParser-->>Sequence: string
    Sequence-->>App: "Why don't Go developers eat sushi? Because they're afraid of goroutine sushi!"
```

---

## Data Flow: Agent Loop

```mermaid
sequenceDiagram
    participant App
    participant AgentExecutor
    participant Agent
    participant ChatModel
    participant Tool

    App->>AgentExecutor: Invoke({"input": "What is 6*7?"})
    loop Until finish or maxIterations
        AgentExecutor->>Agent: Plan(steps, inputs)
        Agent->>ChatModel: Invoke(messages)
        ChatModel-->>Agent: AIMessage{ToolCalls: [{calculator, "6*7"}]}
        Agent-->>AgentExecutor: AgentOutput{Actions: [calculator("6*7")]}
        AgentExecutor->>Tool: Run("6*7")
        Tool-->>AgentExecutor: "42"
        Note over AgentExecutor: Append AgentStep to intermediateSteps
        AgentExecutor->>Agent: Plan(steps=[{calculator, "42"}], inputs)
        Agent->>ChatModel: Invoke(messages with tool result)
        ChatModel-->>Agent: AIMessage{Content: "The answer is 42"}
        Agent-->>AgentExecutor: AgentOutput{Finish: {output: "The answer is 42"}}
    end
    AgentExecutor-->>App: {"output": "The answer is 42"}
```

---

## Data Flow: RAG Pipeline

```mermaid
sequenceDiagram
    participant App
    participant TextSplitter
    participant VectorStore
    participant Embedder
    participant Retriever
    participant RetrievalQA
    participant LLMChain

    Note over App,TextSplitter: Indexing phase (one-time)
    App->>TextSplitter: SplitDocuments(docs)
    TextSplitter-->>App: chunks[]
    App->>VectorStore: AddDocuments(chunks)
    VectorStore->>Embedder: EmbedDocuments(texts)
    Embedder-->>VectorStore: [][]float64 embeddings
    VectorStore-->>App: ids[]

    Note over App,LLMChain: Query phase
    App->>RetrievalQA: Invoke({"query": "When was Go created?"})
    RetrievalQA->>Retriever: GetRelevantDocuments(query)
    Retriever->>VectorStore: SimilaritySearch(query, k)
    VectorStore->>Embedder: EmbedQuery(query)
    Embedder-->>VectorStore: []float64
    VectorStore-->>Retriever: []Document (top-k)
    Retriever-->>RetrievalQA: []Document
    RetrievalQA->>LLMChain: Invoke({query, context})
    LLMChain-->>RetrievalQA: answer string
    RetrievalQA-->>App: answer string
```

---

## Streaming Architecture

```mermaid
graph LR
    Provider["LLM Provider\n(SSE/HTTP stream)"]
    StreamIterator["StreamIterator[AIMessage]"]
    Consumer["Consumer\n(for chunk := range ...)"]

    Provider -->|"channel StreamChunk[T]"| StreamIterator
    StreamIterator -->|"Next() → value, ok, err"| Consumer

    style StreamIterator fill:#f9f,stroke:#333
```

`StreamIterator[T]` wraps an internal Go channel. The provider writes chunks from a goroutine; the consumer calls `Next()` in a pull loop or `Collect()` to gather all chunks at once. Calling `Close()` signals the consumer is done early, preventing goroutine leaks.

---

## Key Abstractions Summary

| Abstraction | Package | Description |
|---|---|---|
| `Runnable[I,O]` | `core` | The universal interface — every component is a `Runnable` |
| `Message` | `core` | Typed message interface: Human, AI, System, Tool, Function |
| `RunnableConfig` | `core` | Per-invocation configuration: tags, metadata, callbacks, stop sequences |
| `CallbackHandler` | `core` | Lifecycle hooks for every event in the pipeline |
| `ChatModel` | `llms` | Extends `Runnable` with `BindTools` and `WithStructuredOutput` |
| `Tool` | `tools` | Named, described function callable by agents |
| `Memory` | `memory` | Load/save conversation context across turns |
| `VectorStore` | `vectorstores` | Embed and similarity-search documents |
| `Embedder` | `embeddings` | Convert text to embedding vectors |
