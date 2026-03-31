# Feature 002: Document Loaders

## User Story

**As a** developer building a RAG pipeline,
**I want** built-in document loaders that can ingest content from files (text, Markdown, PDF, HTML) and URLs,
**so that** I can populate my vector store with real-world data without writing custom parsing code for every source format.

### Acceptance Criteria

- A `DocumentLoader` interface exists with `Load(ctx) ([]*core.Document, error)`.
- Built-in loaders: `TextLoader`, `MarkdownLoader`, `HTMLLoader`, `PDFLoader`, `DirectoryLoader`.
- Each loader sets appropriate metadata on documents (source path, page number for PDFs, title for HTML).
- `DirectoryLoader` recursively loads files from a directory, selecting the right loader by file extension.
- Loaders compose with `TextSplitter` — e.g., `loader.LoadAndSplit(ctx, splitter)`.
- Loaders respect `context.Context` for cancellation (useful for large directories or slow network sources).
- URL-based loaders (`URLLoader`) fetch content over HTTP with configurable timeouts.

### Example Usage

```go
// Load a single text file
loader := loaders.NewTextLoader("data/notes.txt")
docs, err := loader.Load(ctx)

// Load and split a Markdown file
loader := loaders.NewMarkdownLoader("docs/guide.md")
splitter := textsplitters.NewRecursiveCharacterTextSplitter(
    textsplitters.WithChunkSize(500),
)
docs, err := loader.LoadAndSplit(ctx, splitter)

// Load an entire directory
dirLoader := loaders.NewDirectoryLoader("data/",
    loaders.WithGlob("**/*.{txt,md,html}"),
    loaders.WithRecursive(true),
)
docs, err := dirLoader.Load(ctx)

// Load from a URL
urlLoader := loaders.NewURLLoader("https://example.com/article",
    loaders.WithTimeout(10 * time.Second),
)
docs, err := urlLoader.Load(ctx)
```

---

## Implementation Plan

### New Package: `loaders/`

#### Interface: `loaders/loader.go`

```go
type DocumentLoader interface {
    Load(ctx context.Context) ([]*core.Document, error)
}

// Optional helper interface
type SplittableLoader interface {
    DocumentLoader
    LoadAndSplit(ctx context.Context, splitter textsplitters.TextSplitter) ([]*core.Document, error)
}
```

Provide a `BaseLoader` struct that embeds loading logic and implements `LoadAndSplit` generically (load, then split).

#### Implementations

1. **`TextLoader`** (`loaders/text.go`)
   - Reads a file into a single `Document` with `{source: filepath}` metadata.
   - Uses `os.ReadFile`.

2. **`MarkdownLoader`** (`loaders/markdown.go`)
   - Like TextLoader but optionally strips Markdown syntax for cleaner embeddings.
   - Adds `{source, format: "markdown"}` metadata.

3. **`HTMLLoader`** (`loaders/html.go`)
   - Parses HTML, extracts text content (strip tags).
   - Extracts `<title>` into metadata.
   - Uses `golang.org/x/net/html` for parsing.

4. **`PDFLoader`** (`loaders/pdf.go`)
   - Extracts text from PDF files.
   - Returns one `Document` per page with `{source, page: N}` metadata.
   - Uses a pure-Go PDF library (e.g., `github.com/ledongthuc/pdf` or `github.com/dslipak/pdf`).
   - Mark as optional build tag if heavy dependency.

5. **`URLLoader`** (`loaders/url.go`)
   - Fetches content from a URL via `net/http`.
   - Detects content type from `Content-Type` header and delegates to the appropriate parser (text, HTML).
   - Configurable timeout, custom headers, user-agent.

6. **`DirectoryLoader`** (`loaders/directory.go`)
   - Walks a directory tree using `filepath.WalkDir`.
   - Maps file extensions to loaders (`.txt` → TextLoader, `.md` → MarkdownLoader, etc.).
   - Options: `WithGlob`, `WithRecursive`, `WithLoaderMapping(ext → LoaderFactory)`.
   - Collects errors per file without stopping the whole load (returns partial results + multi-error).

### Metadata Convention

All loaders set at minimum:
- `source` — the file path or URL where the document came from.
- Additional keys per loader type (page number, format, title, content-type).

### Testing Strategy

- Unit tests with fixture files in `testdata/` for each format.
- Test `DirectoryLoader` with a temp directory created in tests.
- Test `URLLoader` with `httptest.NewServer`.
- Test `LoadAndSplit` integration with `RecursiveCharacterTextSplitter`.
- Test context cancellation during directory walk.

### Dependencies

- `golang.org/x/net/html` — for HTML parsing (well-established, stdlib-adjacent).
- PDF library — optional, behind a build tag to keep the module lightweight.
