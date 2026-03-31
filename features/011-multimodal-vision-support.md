# Feature 011: Multimodal / Vision Support

## User Story

**As a** developer building applications that process images alongside text,
**I want** to send images (from files, URLs, or base64) as part of my messages to vision-capable models (GPT-4o, Claude, Llama 3.2 Vision),
**so that** I can build applications that analyze images, extract text from screenshots, describe photos, and reason over visual content.

### Acceptance Criteria

- `core.HumanMessage` supports mixed content: text and images in the same message.
- Images can be provided as: local file path, URL, or base64-encoded data.
- The framework automatically converts images to the provider-specific format (OpenAI content blocks, Anthropic image blocks, etc.).
- Image support works transparently with chains, agents, and streaming.
- I can specify image detail level (e.g., OpenAI's `detail: "high"` / `"low"`).
- Unsupported models return a clear error when images are included.

### Example Usage

```go
import "github.com/LucaLanziani/langchain-go/core"

// Image from URL
msg := core.NewHumanMessageWithContent(
    core.TextContent("What's in this image?"),
    core.ImageURLContent("https://example.com/photo.jpg",
        core.WithImageDetail("high"),
    ),
)

// Image from local file (auto base64 encoded)
msg := core.NewHumanMessageWithContent(
    core.TextContent("Describe this screenshot"),
    core.ImageFileContent("./screenshot.png"),
)

// Image from base64
msg := core.NewHumanMessageWithContent(
    core.TextContent("What text is visible?"),
    core.ImageBase64Content(base64Data, "image/png"),
)

// Use in a chain
result, _ := model.Invoke(ctx, []core.Message{msg})
fmt.Println(result.Content) // "The image shows..."

// Use with an agent
prompt := prompts.NewChatPromptTemplate(
    prompts.System("You are a vision assistant. Analyze images the user sends."),
    prompts.Placeholder("agent_scratchpad"),
    prompts.Human("{input}"),
)
// The {input} can contain multimodal messages
```

---

## Implementation Plan

### Core Message Extension: `core/messages.go`

1. **New content types**:

```go
type ContentType string
const (
    ContentTypeText  ContentType = "text"
    ContentTypeImage ContentType = "image"
)

type ContentBlock interface {
    GetContentType() ContentType
}

type TextContent struct {
    Text string
}

type ImageContent struct {
    URL      string // URL or data URI (base64)
    MimeType string // "image/png", "image/jpeg", etc.
    Detail   string // "high", "low", "auto" (provider-specific)
}
```

2. **Extended HumanMessage**:

```go
type HumanMessage struct {
    Content  string         // text-only content (backward compatible)
    Parts    []ContentBlock // multimodal content blocks
    // ...existing fields
}
```

- If `Parts` is non-empty, providers serialize using the multipart format.
- If `Parts` is empty, fall back to `Content` string (backward compatible).

3. **Constructor helpers**:

```go
func NewHumanMessageWithContent(parts ...ContentBlock) *HumanMessage
func TextContent(text string) *textContent
func ImageURLContent(url string, opts ...ImageOption) *ImageContent
func ImageFileContent(path string, opts ...ImageOption) *ImageContent  // reads + base64 encodes
func ImageBase64Content(data string, mimeType string, opts ...ImageOption) *ImageContent
```

### Provider Integration

#### OpenAI (`providers/openai/chat.go`)

Convert `Parts` to OpenAI's content array format:
```json
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,...", "detail": "high"}}
    ]
}
```

#### Anthropic (`providers/anthropic/chat.go`)

Convert to Anthropic's content block format:
```json
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
    ]
}
```

#### Ollama (`providers/ollama/chat.go`)

Convert to Ollama's images field:
```json
{
    "role": "user",
    "content": "What's in this image?",
    "images": ["base64data..."]
}
```

### Backward Compatibility

- `GetContent()` on a multimodal message returns the concatenated text parts.
- Existing code that creates `NewHumanMessage("text")` continues to work unchanged.
- Providers that don't support images return an error: `"model X does not support image inputs"`.

### Testing Strategy

- Unit tests for content block creation (URL, file, base64).
- Unit tests for provider-specific serialization (verify JSON output matches expected format).
- Test backward compatibility: existing text-only messages still work.
- Test `ImageFileContent` reads and encodes a test image correctly.
- Test error when sending images to a text-only model.
- Integration tests with real vision models (behind build tags).

### Dependencies

- No new dependencies. Uses stdlib `encoding/base64`, `os`, `mime`, `path/filepath`.
