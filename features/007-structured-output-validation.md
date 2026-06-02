# Feature 007: Structured Output with Validation

> **GitHub Issue:** [#7](https://github.com/LucaLanziani/langchain-go/issues/7)

## User Story

**As a** developer who needs LLMs to produce structured data (not just free text),
**I want** to define a Go struct as the expected output, have the framework enforce that schema via the model's native structured output or JSON mode, and validate the response automatically,
**so that** I get type-safe, validated data from LLM calls without manual JSON parsing and error handling.

### Acceptance Criteria

- I can call `model.WithStructuredOutput(MyStruct{})` to get a model that returns `MyStruct` instead of `*AIMessage`.
- The schema is automatically generated from the Go struct's tags (`json`, `description`, `required`, `enum`).
- Works with OpenAI's JSON mode / response_format and Anthropic's tool-based structured output.
- If the model returns invalid JSON or a response that doesn't match the schema, a clear validation error is returned.
- Optional retry: on validation failure, automatically re-prompt the model with the error message (configurable max retries).
- A `StructuredOutputParser[T]` can be used standalone in chains (not only through the model wrapper).
- Supports nested structs, arrays, enums (via tags), and optional fields.

### Example Usage

```go
type Movie struct {
    Title    string   `json:"title" description:"The movie title"`
    Year     int      `json:"year" description:"Release year"`
    Genre    string   `json:"genre" enum:"action,comedy,drama,horror,sci-fi"`
    Rating   float64  `json:"rating" description:"Rating from 0 to 10"`
    Cast     []string `json:"cast" description:"Main actors"`
}

// Via model wrapper
typedModel := openai.New().WithStructuredOutput(Movie{})

movie, err := typedModel.Invoke(ctx, []core.Message{
    core.NewHumanMessage("Tell me about Inception"),
})
// movie is of type Movie, fully typed and validated
fmt.Printf("%s (%d) - %.1f\n", movie.Title, movie.Year, movie.Rating)

// Via parser in a chain
parser := outputparsers.NewStructuredOutputParser[Movie](
    outputparsers.WithRetryOnFailure(2), // retry up to 2 times
)

chain := runnable.Pipe3(prompt, model, parser)
movie, err := chain.Invoke(ctx, map[string]any{"query": "Tell me about Inception"})

// Get the JSON schema for prompt injection
schema := parser.GetFormatInstructions()
```

---

## Implementation Plan

### Enhanced Schema Generation: `tools/schema.go` (extend existing)

The existing `generateJSONSchema` in `tools/structured.go` already handles basic Go structs. Extend it:

1. **Enum support** — parse `enum:"val1,val2,val3"` struct tag → `{"enum": ["val1","val2","val3"]}`.
2. **Required fields** — fields without `omitempty` in the `json` tag are required.
3. **Nested structs** — already partially supported; ensure recursive handling with `$defs`.
4. **Slice/array types** — `{"type": "array", "items": {...}}`.
5. **Validation hints** — `min`, `max`, `minLength`, `maxLength` tags.

### Structured Output Parser: `outputparsers/structured.go`

```go
type StructuredOutputParser[T any] struct {
    retryCount int
    model      llms.ChatModel // optional, for retry
    schema     map[string]any
}
```

1. **`Invoke(ctx, input *AIMessage) (T, error)`**:
   - Extract JSON from `input.Content` (handle ```json code blocks).
   - Unmarshal into `T`.
   - Validate against schema (enum checks, required fields, type checks).
   - On failure + retry enabled: re-invoke the model with original messages + error feedback.

2. **`GetFormatInstructions() string`** — return a prompt-ready description of the expected JSON format.

3. **`GetSchema() map[string]any`** — return the JSON schema.

### Validation: `outputparsers/validate.go`

A lightweight JSON Schema validator (subset):
- Type checking (string, number, integer, boolean, array, object).
- Required fields.
- Enum values.
- Min/max for numbers.
- MinLength/maxLength for strings.
- Nested object and array validation.

No need for a full JSON Schema library — validate only the features we generate.

### Model Integration

Enhance `WithStructuredOutput` on each provider:

1. **OpenAI**: Use `response_format: {"type": "json_schema", "json_schema": {...}}` (GPT-4o+) or fall back to `response_format: {"type": "json_object"}` with schema in the system prompt.

2. **Anthropic**: Use tool-based structured output — define a single tool with the schema, force tool use, extract the args as the result.

The `WithStructuredOutput` method returns a `Runnable[[]Message, T]` that wraps the model and parser together.

### Testing Strategy

- Unit tests for schema generation with various struct configurations (nested, arrays, enums, optional).
- Unit tests for validation: valid data passes, invalid data fails with clear errors.
- Unit tests for JSON extraction from various formats (raw JSON, code blocks, mixed text).
- Test retry logic with a mock model that returns invalid JSON first, then valid.
- Test `GetFormatInstructions` produces parseable instructions.
- Integration tests with each provider (behind build tags).

### Dependencies

- No external dependencies. Uses stdlib `encoding/json`, `reflect`, `fmt`, `strings`.
