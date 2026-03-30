# Codebase Reference

Module: `github.com/LucaLanziani/langchain-go` — Go 1.24+.

---

## Package Map

| Import path | Purpose |
|---|---|
| `…/core` | `Runnable[I,O]`, `Message`, `RunnableConfig`, `Option`, `CallbackHandler`, `Document`, `StreamIterator` |
| `…/llms` | `ChatModel` interface, `ToolDefinition`, `ChatResult`, `ChatGeneration`, `TokenUsage` |
| `…/providers/openai` | OpenAI GPT models + embeddings |
| `…/providers/anthropic` | Anthropic Claude models |
| `…/providers/github-copilot` | GitHub Copilot models (requires CLI) |
| `…/prompts` | `ChatPromptTemplate`, `PromptTemplate`, `MessagesPlaceholder` |
| `…/chains` | `LLMChain`, `StuffDocumentsChain`, `RetrievalQA` |
| `…/runnable` | `Sequence`, `Branch`, `Parallel`, `Lambda`, `Passthrough`, `Assign` |
| `…/agents` | `ReActAgent`, `ToolCallingAgent`, `AgentExecutor` |
| `…/tools` | `Tool` interface, `StructuredTool`, `NewTool`, `NewTypedTool[T]` |
| `…/memory` | `Memory` interface, `ConversationBufferMemory`, `ConversationWindowMemory`, `ChatMessageHistory` |
| `…/embeddings` | `Embedder` interface |
| `…/vectorstores` | `VectorStore` interface, `DocumentWithScore` |
| `…/vectorstores/inmemory` | In-memory cosine-similarity vector store |
| `…/retrievers` | `Retriever` interface, `VectorStoreRetriever` |
| `…/textsplitters` | `RecursiveCharacterTextSplitter` |
| `…/outputparsers` | `StringOutputParser`, `JSONOutputParser` |
| `…/callbacks` | `Manager`, `StdoutHandler`, `LangSmithHandler` |

---

## Core Interface

```go
// core/runnable.go
type Runnable[I, O any] interface {
    Invoke(ctx context.Context, input I, opts ...Option) (O, error)
    Stream(ctx context.Context, input I, opts ...Option) (*StreamIterator[O], error)
    Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error)
    GetName() string
}
```

**RULE:** Every component is a `Runnable`. Compose via `runnable.Pipe2/3/4`, `runnable.NewSequence`, or the chain/agent wrappers.

---

## Message Types

```go
// core/messages.go
type Message interface { GetType() MessageType; GetContent() string; GetName() string; GetAdditionalKwargs() map[string]any }

// Concrete types (all embed BaseMessage):
core.NewHumanMessage(content string) *HumanMessage
core.NewAIMessage(content string) *AIMessage
core.NewAIMessageWithToolCalls(content string, toolCalls []ToolCall) *AIMessage
core.NewSystemMessage(content string) *SystemMessage
core.NewToolMessage(content, toolCallID string) *ToolMessage

// AIMessage extras:
type AIMessage struct {
    BaseMessage
    ToolCalls      []ToolCall      // when model requests tool invocations
    ToolCallChunks []ToolCallChunk // during streaming
    UsageMetadata  *UsageMetadata  // input/output/total tokens
}

type ToolCall struct {
    ID   string          `json:"id"`
    Name string          `json:"name"`
    Args json.RawMessage `json:"args"` // raw JSON from model
}
```

---

## Configuration & Options

```go
// core/config.go — pass to any Invoke/Stream/Batch call
core.WithTags("tag1", "tag2")
core.WithMetadata(map[string]any{"key": "val"})
core.WithCallbacks(handler1, handler2)
core.WithRunName("my-run")
core.WithMaxConcurrency(8)
core.WithRecursionLimit(10)
core.WithRunID("my-uuid")
core.WithStop("\nObservation:")
core.WithConfigurable(map[string]any{"model": "gpt-4o-mini"})

// llms/options.go — model inference options, also accepted as core.Option
llms.WithTemperature(0.7)
llms.WithMaxTokens(1024)
llms.WithTopP(0.9)
llms.WithModel("gpt-4o-mini")
```

---

## ChatModel Interface

```go
// llms/chatmodel.go
type ChatModel interface {
    core.Runnable[[]core.Message, *core.AIMessage]
    Generate(ctx context.Context, messageSets [][]core.Message, opts ...core.Option) (*ChatResult, error)
    BindTools(tools ...ToolDefinition) ChatModel      // returns new bound model
    WithStructuredOutput(schema map[string]any) ChatModel
}
```

### Provider Construction

```go
// OpenAI — reads OPENAI_API_KEY
openai.New(...openai.OptionFunc) *ChatModel
openai.New(openai.WithAPIKey("sk-..."), openai.WithModelName("gpt-4o-mini"))

// Anthropic — reads ANTHROPIC_API_KEY
anthropic.New(...anthropic.OptionFunc) *ChatModel
anthropic.New(anthropic.WithModelName("claude-3-haiku-20240307"), anthropic.WithMaxTokens(1024))

// GitHub Copilot — reads GITHUB_TOKEN; must call Close()
copilot.New(...copilot.OptionFunc) (*ChatModel, error)
defer model.Close()

// Embeddings (OpenAI only)
openai.NewEmbeddings(...openai.OptionFunc) *Embeddings
```

### Default Models

| Provider | Default model |
|---|---|
| OpenAI | `gpt-4o` |
| Anthropic | `claude-sonnet-4-20250514` (maxTokens=4096) |
| GitHub Copilot | `gpt-5-mini` |

---

## Prompt Templates

```go
// prompts/chat.go
prompts.NewChatPromptTemplate(templates ...MessageTemplate) *ChatPromptTemplate
prompts.FromMessages(messages []MessageTemplate) *ChatPromptTemplate

// MessageTemplate constructors:
prompts.System("You are {role}.")      // SystemMessage template
prompts.Human("{question}")            // HumanMessage template
prompts.AI("{response}")               // AIMessage template
prompts.Placeholder("key")            // injects inputs["key"] ([]core.Message)

// ChatPromptTemplate implements Runnable[map[string]any, []core.Message]
messages, err := prompt.FormatMessages(map[string]any{"role": "assistant", "question": "..."})
```

Template variable syntax: `{variable_name}` (no spaces).

---

## Runnable Combinators

```go
// runnable/sequence.go — type-safe linear pipelines
runnable.Pipe2(a Runnable[I,M], b Runnable[M,O]) *Sequence[I,O]
runnable.Pipe3(a, b, c) *Sequence[I,O]
runnable.Pipe4(a, b, c, d) *Sequence[I,O]
runnable.Pipe(steps ...any) *Sequence[any,any]  // variadic, loses type parameters

// runnable/branch.go — conditional routing
runnable.NewBranch[I,O](conditions []BranchCondition[I,O], defaultBranch Runnable[I,O]) *Branch[I,O]
// BranchCondition{Condition: func(I) bool, Runnable: Runnable[I,O]}

// runnable/parallel.go — fan-out with same input
runnable.NewParallel[I,O](branches map[string]Runnable[I,O]) *Parallel[I]      // homogeneous output
runnable.NewParallelAny[I](branches map[string]func(ctx, I, ...Option)(any,error)) *Parallel[I]

// runnable/lambda.go — wrap any function
runnable.NewLambda[I,O](fn func(context.Context, I)(O, error)) *Lambda[I,O]

// runnable/passthrough.go — identity
runnable.NewPassthrough[T]() *Passthrough[T]

// runnable/passthrough.go — augment a map
runnable.NewAssign[I](additions map[string]func(ctx,I,...Option)(any,error)) *Assign[I]
```

All primitives implement `Runnable` and have optional `.WithName(string)` for tracing.

---

## Predefined Chains

```go
// chains/chains.go
chains.NewLLMChain(llm ChatModel, prompt *ChatPromptTemplate) *LLMChain
// Runnable[map[string]any, string]

chains.NewStuffDocumentsChain(llmChain *LLMChain, contextKey string) *StuffDocumentsChain
// Runnable[[]*core.Document, string]

chains.NewRetrievalQA(retriever Retriever, llmChain *LLMChain) *RetrievalQA
// Runnable[map[string]any, string] — expects input key "query"
```

---

## Output Parsers

```go
// outputparsers/string.go — Runnable[*core.AIMessage, string]
outputparsers.NewStringOutputParser() *StringOutputParser

// outputparsers/json.go — Runnable[*core.AIMessage, map[string]any]
outputparsers.NewJSONOutputParser() *JSONOutputParser
```

---

## Tools

```go
// tools/tool.go
type Tool interface {
    Name() string
    Description() string
    ArgsSchema() map[string]any  // JSON Schema
    Run(ctx context.Context, input string) (string, error)
}

// tools/structured.go
tools.NewTool(name, description string, fn func(context.Context, string)(string,error)) *StructuredTool
tools.NewTypedTool[T any](name, description string, argsExample T, fn func(context.Context, T)(string,error)) *StructuredTool

// Conversions
tools.ToDefinition(t Tool) llms.ToolDefinition
tools.ToDefinitions(tools ...Tool) []llms.ToolDefinition
tools.NewRunnableTool(t Tool) *RunnableTool  // Runnable[string,string]

// Execution helpers
tools.ExecuteToolCall(ctx, toolMap map[string]Tool, tc core.ToolCall) (string, error)
tools.ExecuteToolCalls(ctx, toolMap map[string]Tool, tcs []core.ToolCall) ([]string, error)
tools.ParseToolCallArgs(tc core.ToolCall, dest any) error
```

**Typed tool example:**
```go
type Args struct {
    Query   string `json:"query"   description:"The search query"`
    MaxHits int    `json:"max_hits" description:"Max results to return"`
}
tool := tools.NewTypedTool("search", "Search the web.", Args{},
    func(_ context.Context, a Args) (string, error) { ... },
)
```

---

## Agents

```go
// agents/types.go
type AgentAction struct { Tool, ToolInput, Log string; MessageLog []core.Message }
type AgentFinish  struct { ReturnValues map[string]any; Log string; MessageLog []core.Message }
type AgentStep    struct { Action AgentAction; Observation string }
type AgentOutput  struct { Actions []AgentAction; Finish *AgentFinish } // one of the two fields is set

// agents/executor.go
type Agent interface {
    Plan(ctx, intermediateSteps []AgentStep, inputs map[string]any) (*AgentOutput, error)
    InputKeys() []string
    OutputKeys() []string
}

agents.NewToolCallingAgent(llm ChatModel, tools []Tool, prompt *ChatPromptTemplate) *ToolCallingAgent
agents.NewReActAgent(llm ChatModel, tools []Tool, prompt *ChatPromptTemplate) *ReActAgent  // nil prompt = default

agents.NewAgentExecutor(agent Agent, tools []Tool, ...ExecutorOption) *AgentExecutor
// Options:
agents.WithMaxIterations(15)
agents.WithReturnIntermediateSteps(true)
agents.WithHandleParsingErrors(true)
```

**Executor output keys:** `"output"` always; `"intermediate_steps"` when `WithReturnIntermediateSteps(true)`.

**Prompt requirement for both agent types:** must include `prompts.Placeholder("agent_scratchpad")`.

**ToolCallingAgent vs ReActAgent:** prefer `ToolCallingAgent` — it uses native model tool calling and is more reliable. Use `ReActAgent` only for models without tool calling support.

---

## Memory

```go
// memory/memory.go
type Memory interface {
    MemoryVariables() []string
    LoadMemoryVariables(ctx, inputs map[string]any) (map[string]any, error)
    SaveContext(ctx, inputs, outputs map[string]any) error
    Clear(ctx) error
}

memory.NewConversationBufferMemory() *ConversationBufferMemory
// Fields: MemoryKey="history", InputKey="input", OutputKey="output",
//         ReturnMessages=false, HumanPrefix="Human", AIPrefix="AI", MaxMessages=0

memory.NewConversationWindowMemory(k int) *ConversationWindowMemory
// Keeps last k turns (k*2 messages). Same fields as Buffer.

memory.NewChatMessageHistory() *ChatMessageHistory
// Thread-safe: AddMessage, AddUserMessage, AddAIMessage, GetMessages, SetMessages, Clear
```

Memory is **not** automatically wired into chains — you must call `LoadMemoryVariables` before `Invoke` and `SaveContext` after.

---

## RAG Stack

```go
// embeddings/embeddings.go
type Embedder interface {
    EmbedDocuments(ctx, texts []string) ([][]float64, error)
    EmbedQuery(ctx, text string) ([]float64, error)
}
openai.NewEmbeddings(...openai.OptionFunc) *Embeddings  // default model: text-embedding-ada-002

// vectorstores/vectorstore.go
type VectorStore interface {
    AddDocuments(ctx, docs []*core.Document) ([]string, error)
    SimilaritySearch(ctx, query string, k int) ([]*core.Document, error)
    SimilaritySearchWithScore(ctx, query string, k int) ([]DocumentWithScore, error)
    Delete(ctx, ids []string) error
    GetEmbedder() Embedder
}
inmemory.New(embedder Embedder) *Store  // cosine similarity, not persistent

// retrievers/retriever.go — Runnable[string, []*core.Document]
retrievers.NewVectorStoreRetriever(store VectorStore, k int) *VectorStoreRetriever
// k <= 0 defaults to 4

// textsplitters/recursive.go
textsplitters.NewRecursiveCharacterTextSplitter(chunkSize, chunkOverlap int) *RecursiveCharacterTextSplitter
// Default separators: ["\n\n", "\n", " ", ""]
// .WithSeparators([]string{...}) to override
// .SplitText(string) []string
// .SplitDocuments([]*Document) []*Document  — preserves metadata

// core/documents.go
core.NewDocument(pageContent string) *Document
// Fields: PageContent, Metadata map[string]any, ID string
```

---

## Callbacks

```go
// core/callbacks.go
type CallbackHandler interface {
    OnLLMStart(ctx, prompts []string, runID, parentRunID string, extra map[string]any)
    OnChatModelStart(ctx, messages []Message, runID, parentRunID string, extra map[string]any)
    OnLLMNewToken(ctx, token, runID string)
    OnLLMEnd(ctx, result LLMResult, runID string)
    OnLLMError(ctx, err, runID string)
    OnChainStart(ctx, inputs map[string]any, runID, parentRunID string, extra map[string]any)
    OnChainEnd(ctx, outputs map[string]any, runID string)
    OnChainError(ctx, err, runID string)
    OnToolStart(ctx, tool, input, runID, parentRunID string, extra map[string]any)
    OnToolEnd(ctx, output, runID string)
    OnToolError(ctx, err, runID string)
    OnAgentAction(ctx, action AgentActionData, runID string)
    OnAgentFinish(ctx, finish AgentFinishData, runID string)
    OnRetrieverStart(ctx, query, runID, parentRunID string, extra map[string]any)
    OnRetrieverEnd(ctx, documents any, runID string)
    OnText(ctx, text string)
}
// Embed core.BaseCallbackHandler for no-op defaults; override only what you need.

// Built-in handlers:
callbacks.NewStdoutHandler() *StdoutHandler       // color ANSI output, debugging
callbacks.NewLangSmithHandler(project string) *LangSmithHandler
// Reads LANGCHAIN_API_KEY, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT from env.

callbacks.NewManager(handlers ...CallbackHandler) *Manager
// .WithInheritableHandlers(...) .GetChild(tag) .AllHandlers()
```

---

## Streaming Pattern

```go
stream, err := model.Stream(ctx, messages)
if err != nil { return err }
defer stream.Close() // always close to prevent goroutine leaks

for {
    chunk, ok, err := stream.Next()
    if err != nil  { return err }
    if !ok         { break }
    process(chunk)
}
// or: chunks, err := stream.Collect()
```

---

## Common Patterns

### Minimal chat
```go
model := openai.New()
resp, err := model.Invoke(ctx, []core.Message{core.NewHumanMessage("Hi!")})
```

### Typed pipeline
```go
chain := runnable.Pipe3(
    prompts.NewChatPromptTemplate(prompts.Human("{input}")),
    openai.New(),
    outputparsers.NewStringOutputParser(),
)
result, err := chain.Invoke(ctx, map[string]any{"input": "Hello"})
```

### Agent
```go
agent := agents.NewToolCallingAgent(openai.New(), toolList,
    prompts.NewChatPromptTemplate(
        prompts.System("Use tools as needed."),
        prompts.Placeholder("agent_scratchpad"),
        prompts.Human("{input}"),
    ),
)
exec := agents.NewAgentExecutor(agent, toolList)
out, err := exec.Invoke(ctx, map[string]any{"input": "..."})
// out["output"] = final answer string
```

### RAG
```go
store := inmemory.New(openai.NewEmbeddings())
store.AddDocuments(ctx, chunks)
chain := chains.NewRetrievalQA(
    retrievers.NewVectorStoreRetriever(store, 4),
    chains.NewLLMChain(openai.New(), qaPrompt),
)
answer, err := chain.Invoke(ctx, map[string]any{"query": "..."})
```

### Memory loop
```go
mem := memory.NewConversationBufferMemory()
mem.ReturnMessages = true

for _, input := range turns {
    vars, _ := mem.LoadMemoryVariables(ctx, nil)
    out, _ := chain.Invoke(ctx, map[string]any{"input": input, "history": vars["history"]})
    mem.SaveContext(ctx, map[string]any{"input": input}, map[string]any{"output": out})
}
```

---

## Coding Conventions

- **Functional options pattern** everywhere: `New(...OptionFunc)`. Never set fields directly on provider structs.
- **`core.Option` vs provider `OptionFunc`**: `core.Option` is for per-call config (tags, callbacks, stop); provider `OptionFunc` is for long-lived model config (API key, model name).
- **Error wrapping**: use `fmt.Errorf("context: %w", err)`.
- **Context threading**: always thread `ctx context.Context` as the first parameter and pass it to sub-calls.
- **No global state**: no init() side effects, no package-level vars that accumulate state.
- **Generics**: prefer typed `Pipe2/3/4` over `Pipe` to keep compile-time type safety.
- **Test files** live beside production files (`*_test.go` in same package).

---

## Test Commands

```bash
go test ./...                          # all packages
go test ./runnable/... -run TestSeq    # single test
go test -v ./agents/...                # verbose
go build ./...                         # verify compilation
```
