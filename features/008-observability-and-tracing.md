# Feature 008: Observability with OpenTelemetry and LangSmith

## User Story

**As a** developer running AI applications in production,
**I want** integrated observability that traces every LLM call, tool execution, and chain step with timing, token usage, and cost data,
**so that** I can debug failures, monitor performance, track spending, and optimize my AI pipelines using standard observability tools (Jaeger, Datadog, Grafana) or LangSmith.

### Acceptance Criteria

- An OpenTelemetry callback handler emits spans for every Runnable step (LLM calls, tool executions, chain steps, retriever queries).
- Spans include attributes: model name, token counts (prompt/completion/total), latency, input/output content (configurable), error details.
- Spans are nested correctly: an agent executor span contains child spans for each Plan → Tool → Observe cycle.
- A LangSmith callback handler sends traces to the LangSmith API for the LangSmith UI.
- A cost tracker callback computes estimated cost per call based on model and token counts.
- All handlers are composable — I can use multiple simultaneously.
- Sensitive content (API keys, PII) can be redacted from traces via configuration.

### Example Usage

```go
import (
    "github.com/LucaLanziani/langchain-go/callbacks"
    "go.opentelemetry.io/otel"
)

// OpenTelemetry tracing
tracer := otel.Tracer("my-app")
otelHandler := callbacks.NewOpenTelemetryHandler(tracer,
    callbacks.WithContentRecording(true),  // include input/output in spans
    callbacks.WithRedactPatterns([]string{`sk-[a-zA-Z0-9]+`}), // redact API keys
)

// LangSmith tracing
langsmithHandler := callbacks.NewLangSmithHandler(
    callbacks.WithLangSmithAPIKey("ls-..."),
    callbacks.WithLangSmithProject("my-project"),
)

// Cost tracking
costHandler := callbacks.NewCostTracker()

// Compose all handlers
model := openai.New()
result, err := model.Invoke(ctx, messages,
    core.WithCallbacks(otelHandler, langsmithHandler, costHandler),
)

// Query accumulated costs
fmt.Printf("Total cost: $%.4f\n", costHandler.TotalCost())
fmt.Printf("Total tokens: %d\n", costHandler.TotalTokens())
```

---

## Implementation Plan

### OpenTelemetry Handler: `callbacks/otel.go`

1. **`OpenTelemetryHandler`** — implements `core.CallbackHandler`:
   - `OnLLMStart`: create a span `llm.{model_name}`, set attributes: `llm.model`, `llm.provider`, `llm.input` (if content recording enabled).
   - `OnLLMEnd`: end the span, add attributes: `llm.output`, `llm.tokens.prompt`, `llm.tokens.completion`, `llm.tokens.total`.
   - `OnLLMError`: record error on span, set span status to Error.
   - `OnToolStart`: create child span `tool.{tool_name}`, attributes: `tool.name`, `tool.input`.
   - `OnToolEnd`: end span, add `tool.output`.
   - `OnChainStart` / `OnChainEnd`: create/end span `chain.{chain_name}`.
   - `OnAgentAction`: create span `agent.action.{tool_name}`.
   - `OnAgentFinish`: end agent span.
   - `OnRetrieverStart` / `OnRetrieverEnd`: create spans for retrieval.

2. **Span context propagation** — use `context.Context` to nest spans. The `Manager.GetChild()` pattern already creates child contexts; leverage this for span parenting.

3. **Configuration options**:
   - `WithContentRecording(bool)` — include input/output text in span attributes (default: false for privacy).
   - `WithRedactPatterns([]string)` — regex patterns to redact from recorded content.
   - `WithSpanNamePrefix(string)` — customize span name prefix.

### LangSmith Handler: `callbacks/langsmith.go` (complete the existing stub)

1. **`LangSmithHandler`** — sends run data to LangSmith API:
   - On start events: POST to `https://api.smith.langchain.com/runs` with `{run_type, name, inputs, start_time, parent_run_id}`.
   - On end events: PATCH to update with `{outputs, end_time, status}`.
   - On error events: PATCH with `{error, status: "error"}`.
   - Buffer and batch API calls asynchronously (background goroutine with flush interval).

2. **Configuration**:
   - `WithLangSmithAPIKey` / env var `LANGSMITH_API_KEY`.
   - `WithLangSmithProject` / env var `LANGSMITH_PROJECT`.
   - `WithLangSmithEndpoint` (default: `https://api.smith.langchain.com`).
   - `WithBatchSize(int)` — flush every N runs (default: 10).
   - `WithFlushInterval(time.Duration)` — flush every N seconds (default: 5s).

### Cost Tracker: `callbacks/cost.go`

1. **`CostTracker`** — implements `core.CallbackHandler`:
   - Maintains a thread-safe accumulator of token counts and costs.
   - On `OnLLMEnd`: extract `TokenUsage` from the result, look up per-token cost for the model, accumulate.
   - Model pricing table (embedded, user-overridable):
     ```go
     var defaultPricing = map[string]ModelCost{
         "gpt-4o":        {Input: 0.0025, Output: 0.01},   // per 1K tokens
         "gpt-4":         {Input: 0.03,   Output: 0.06},
         "claude-sonnet-4-20250514": {Input: 0.003, Output: 0.015},
         // ...
     }
     ```
   - Methods: `TotalCost() float64`, `TotalTokens() int`, `CostByModel() map[string]float64`, `Reset()`.

### Testing Strategy

- Unit tests for OpenTelemetry handler using `go.opentelemetry.io/otel/sdk/trace/tracetest` (in-memory exporter).
- Verify span hierarchy: agent → plan → llm, agent → tool.
- Verify attributes are set correctly on each span type.
- Unit tests for LangSmith handler with `httptest.NewServer`.
- Unit tests for cost tracker with known token counts and pricing.
- Test redaction patterns remove sensitive content.
- Test concurrent access to cost tracker.

### Dependencies

- `go.opentelemetry.io/otel` and `go.opentelemetry.io/otel/trace` — for OpenTelemetry integration.
- OpenTelemetry handler should be a separate sub-module to avoid pulling OTel into the core.
- LangSmith and cost tracker use only stdlib (`net/http`, `encoding/json`, `sync`).
