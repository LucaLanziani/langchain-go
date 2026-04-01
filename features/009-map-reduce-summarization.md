# Feature 009: Map-Reduce Summarization Chain

> **GitHub Issue:** [#9](https://github.com/LucaLanziani/langchain-go/issues/9)

## User Story

**As a** developer working with large documents that exceed an LLM's context window,
**I want** a map-reduce summarization chain that splits a document into chunks, summarizes each chunk in parallel, then combines the summaries into a final summary,
**so that** I can summarize arbitrarily long documents efficiently without hitting token limits.

### Acceptance Criteria

- A `MapReduceChain` splits input documents into chunks, runs a "map" chain on each chunk in parallel, then runs a "reduce" chain on the combined map outputs.
- Built-in `SummarizationChain` wraps MapReduceChain with sensible defaults for summarization.
- I can customize the map prompt (per-chunk summary) and reduce prompt (combine summaries).
- Parallel map execution respects `MaxConcurrency` for rate limiting.
- If the combined map outputs still exceed the reduce step's context window, the reduce step iterates (hierarchical reduction) until the output fits.
- The chain streams progress through callbacks (map step N/M completed, reduce step started).
- Works with any `ChatModel` and `TextSplitter`.

### Example Usage

```go
import "github.com/LucaLanziani/langchain-go/chains"

model := openai.New()

// Quick summarization with defaults
summarizer := chains.NewSummarizationChain(model,
    chains.WithChunkSize(4000),
    chains.WithChunkOverlap(200),
)

docs := []*core.Document{{PageContent: veryLongText}}
result, err := summarizer.Invoke(ctx, map[string]any{
    "input_documents": docs,
})
fmt.Println(result["output_text"])

// Custom map-reduce with user-defined prompts
mapPrompt := prompts.NewChatPromptTemplate(
    prompts.System("Summarize the following text in 3 bullet points."),
    prompts.Human("{text}"),
)
reducePrompt := prompts.NewChatPromptTemplate(
    prompts.System("Combine these summaries into a single coherent summary."),
    prompts.Human("{text}"),
)

mapReduce := chains.NewMapReduceChain(model,
    chains.WithMapPrompt(mapPrompt),
    chains.WithReducePrompt(reducePrompt),
    chains.WithMaxConcurrency(5),
    chains.WithMaxReduceIterations(3),
)

result, err := mapReduce.Invoke(ctx, map[string]any{
    "input_documents": docs,
})
```

---

## Implementation Plan

### New Chain: `chains/mapreduce.go`

#### `MapReduceChain`

```go
type MapReduceChain struct {
    model               llms.ChatModel
    mapPrompt           *prompts.ChatPromptTemplate
    reducePrompt        *prompts.ChatPromptTemplate
    splitter            textsplitters.TextSplitter
    maxConcurrency      int
    maxReduceIterations int
    collapsePrompt      *prompts.ChatPromptTemplate // optional, for intermediate reduce
    tokenCounter        func(string) int            // estimate tokens
    maxReduceTokens     int                         // threshold for triggering iterative reduce
}
```

#### Invoke Flow

1. **Split**: If input is raw text or documents, split using the configured `TextSplitter`.
2. **Map**: For each chunk, run `mapPrompt → model → StringOutputParser` in parallel (bounded by `MaxConcurrency`). Fire `OnChainStart` callback per chunk.
3. **Combine**: Join all map outputs into a single text.
4. **Check fit**: Estimate tokens of the combined text. If over `maxReduceTokens`, do an intermediate "collapse" step:
   - Group map outputs into batches that fit within the token limit.
   - Run the collapse/reduce prompt on each batch.
   - Repeat until the combined output fits.
5. **Reduce**: Run `reducePrompt → model → StringOutputParser` on the final combined text.
6. **Return**: `map[string]any{"output_text": finalSummary}`.

#### Token Estimation

- Default: `len(text) / 4` (rough approximation of tokens for English text).
- Configurable via `WithTokenCounter(fn func(string) int)`.
- Optional: integrate with `tiktoken-go` for accurate counting.

### Convenience Wrapper: `chains/summarize.go`

```go
func NewSummarizationChain(model llms.ChatModel, opts ...SummarizeOption) *MapReduceChain {
    // Default map prompt: "Write a concise summary of the following: {text}"
    // Default reduce prompt: "Write a concise summary of these summaries: {text}"
    // Default splitter: RecursiveCharacterTextSplitter with chunk size 4000
    return NewMapReduceChain(model, defaultMapPrompt, defaultReducePrompt, defaults...)
}
```

### Streaming Support

- The chain fires callbacks during the map phase: `OnMapStepComplete(index, total, summary)`.
- The final reduce step can stream via `model.Stream()`.
- Progress can be tracked through the callback manager.

### Testing Strategy

- Unit test: map phase runs in parallel and produces per-chunk summaries (mock model).
- Unit test: reduce phase combines summaries correctly.
- Unit test: hierarchical reduce triggers when combined output is too large.
- Unit test: MaxConcurrency is respected (track timing of mock model calls).
- Unit test: context cancellation stops in-progress map steps.
- Test with a document that requires exactly 1 chunk (no map-reduce, just summarize).

### Dependencies

- No new dependencies. Uses existing `textsplitters`, `prompts`, `outputparsers` packages.
