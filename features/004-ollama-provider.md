# Feature 004: Ollama Provider for Local LLMs

## User Story

**As a** developer who wants to run AI applications locally without cloud API dependencies,
**I want** an Ollama provider that connects to a local Ollama instance,
**so that** I can develop, test, and run langchain-go applications using open-source models (Llama 3, Mistral, Phi, Gemma, etc.) without API keys or internet access.

### Acceptance Criteria

- An Ollama provider implements `llms.ChatModel` with full `Invoke`, `Stream`, and `Batch` support.
- The provider connects to Ollama's HTTP API (default `http://localhost:11434`).
- Tool calling works for models that support it (Llama 3.1+, Mistral, etc.).
- Streaming uses Ollama's streaming endpoint and returns a proper `StreamIterator`.
- I can configure the base URL, model name, and generation parameters (temperature, top_p, num_predict, etc.).
- The provider provides an `Embedder` implementation using Ollama's embedding endpoint.
- Works as a drop-in replacement for OpenAI in chains and agents (same `Runnable` interface).
- No CGo or native dependencies — pure HTTP client.

### Example Usage

```go
import "github.com/LucaLanziani/langchain-go/providers/ollama"

// Chat model
model := ollama.New(
    ollama.WithModel("llama3.1"),
    ollama.WithBaseURL("http://localhost:11434"), // default
    ollama.WithTemperature(0.7),
)

// Use in a chain (identical to OpenAI usage)
chain := runnable.Pipe3(prompt, model, parser)
result, err := chain.Invoke(ctx, input)

// Streaming
stream, _ := model.Stream(ctx, messages)
for {
    chunk, ok, err := stream.Next()
    if !ok { break }
    fmt.Print(chunk.Content)
}

// Tool calling with supported models
agent := agents.NewToolCallingAgent(model, myTools, prompt)

// Embeddings
embedder := ollama.NewEmbeddings(
    ollama.WithModel("nomic-embed-text"),
)
vectors, _ := embedder.EmbedDocuments(ctx, texts)
```

---

## Implementation Plan

### New Package: `providers/ollama/`

#### Options: `providers/ollama/options.go`

```go
type options struct {
    BaseURL     string  // default: "http://localhost:11434"
    Model       string  // default: "llama3.1"
    Temperature float64
    TopP        float64
    TopK        int
    NumPredict  int // max tokens
    Stop        []string
    NumCtx      int // context window size
    Format      string // "json" or ""
    KeepAlive   string // model keep-alive duration
}
```

Functional options: `WithModel`, `WithBaseURL`, `WithTemperature`, `WithTopP`, `WithTopK`, `WithNumPredict`, `WithStop`, `WithNumCtx`, `WithFormat`, `WithKeepAlive`.

#### Chat Model: `providers/ollama/chat.go`

1. **`ChatModel`** struct — holds `options` and `*http.Client`.

2. **`Invoke`** — POST to `/api/chat`:
   - Convert `[]core.Message` → Ollama message format (`role`, `content`, `images`, `tool_calls`).
   - Map system/human/ai/tool messages to Ollama's `system`/`user`/`assistant`/`tool` roles.
   - Include `tools` array when tools are bound (via `BindTools`).
   - Parse response → `*core.AIMessage` with content, tool calls, usage metadata.

3. **`Stream`** — POST to `/api/chat` with `"stream": true`:
   - Read NDJSON (newline-delimited JSON) response line by line.
   - Each line is a partial response; push to `StreamIterator` channel.
   - Final line has `"done": true` with total token counts.

4. **`Batch`** — parallel Invoke calls with `MaxConcurrency` from config.

5. **`Generate`** — wraps Invoke to return `*llms.ChatResult`.

6. **`BindTools`** — return a new ChatModel with tools attached (Ollama supports OpenAI-compatible tool format for supported models).

7. **`WithStructuredOutput`** — set `format: "json"` and include schema in system prompt.

#### Embeddings: `providers/ollama/embeddings.go`

1. **`Embeddings`** struct — holds options and `*http.Client`.

2. **`EmbedDocuments`** — POST to `/api/embed` (batch endpoint, Ollama 0.4+) or fall back to `/api/embeddings` (single).

3. **`EmbedQuery`** — single text embedding.

#### Message Conversion: `providers/ollama/convert.go`

- Map `core.HumanMessage` → `{role: "user", content: "..."}`
- Map `core.AIMessage` → `{role: "assistant", content: "...", tool_calls: [...]}`
- Map `core.SystemMessage` → `{role: "system", content: "..."}`
- Map `core.ToolMessage` → `{role: "tool", content: "..."}`

#### Ollama API Types: `providers/ollama/types.go`

Internal types matching Ollama's JSON API schema (request/response structs).

### Testing Strategy

- Unit tests with `httptest.NewServer` mocking Ollama's API responses.
- Test message conversion for all message types.
- Test streaming with simulated NDJSON responses.
- Test tool calling flow (bind tools → invoke → parse tool calls).
- Test embeddings endpoint.
- Integration test (skipped in CI) against a real Ollama instance.

### Dependencies

- No external dependencies. Uses stdlib `net/http`, `encoding/json`, `bufio`.
