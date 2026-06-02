# Feature 005: PostgreSQL pgvector Store

> **GitHub Issue:** [#5](https://github.com/LucaLanziani/langchain-go/issues/5)

## User Story

**As a** developer building a production RAG application,
**I want** a vector store backed by PostgreSQL with pgvector,
**so that** I can persist, index, and query document embeddings using battle-tested infrastructure I already operate, without adding a specialized vector database to my stack.

### Acceptance Criteria

- A `pgvector.Store` implements the `vectorstores.VectorStore` interface.
- I can add documents with embeddings and query by similarity (cosine, L2, inner product).
- Documents are persisted in PostgreSQL and survive application restarts.
- The store handles table creation and pgvector extension setup automatically (with opt-out).
- I can configure the table name, embedding dimension, distance metric, and connection pool.
- Similarity search supports filtering by metadata using PostgreSQL JSONB operators.
- The store supports the full interface: `AddDocuments`, `SimilaritySearch`, `SimilaritySearchWithScore`, `Delete`.
- Works with any `embeddings.Embedder` (OpenAI or other compatible implementations).
- Connection is managed via `database/sql` + a PostgreSQL driver — no ORM dependency.

### Example Usage

```go
import (
    "github.com/LucaLanziani/langchain-go/vectorstores/pgvector"
    "github.com/LucaLanziani/langchain-go/providers/openai"
)

embedder := openai.NewEmbeddings()

store, err := pgvector.New(ctx,
    pgvector.WithConnectionString("postgres://user:pass@localhost:5432/mydb"),
    pgvector.WithEmbedder(embedder),
    pgvector.WithTableName("documents"),         // default: "langchain_documents"
    pgvector.WithEmbeddingDimension(1536),
    pgvector.WithDistanceMetric(pgvector.Cosine), // default
)
if err != nil { log.Fatal(err) }
defer store.Close()

// Add documents
docs := []*core.Document{
    {PageContent: "Go is a statically typed language.", Metadata: map[string]any{"source": "wiki"}},
    {PageContent: "Rust focuses on memory safety.", Metadata: map[string]any{"source": "wiki"}},
}
ids, err := store.AddDocuments(ctx, docs)

// Similarity search
results, err := store.SimilaritySearch(ctx, "type-safe programming", 5)

// Search with metadata filter
results, err := store.SimilaritySearch(ctx, "programming languages", 5,
    pgvector.WithFilter(map[string]any{"source": "wiki"}),
)

// Search with scores
scored, err := store.SimilaritySearchWithScore(ctx, "memory safety", 3)
for _, ds := range scored {
    fmt.Printf("%.3f: %s\n", ds.Score, ds.Document.PageContent)
}

// Delete by IDs
err = store.Delete(ctx, ids[:1])
```

---

## Implementation Plan

### New Package: `vectorstores/pgvector/`

#### Options: `vectorstores/pgvector/options.go`

```go
type options struct {
    ConnectionString   string
    DB                 *sql.DB // or provide an existing connection
    TableName          string  // default: "langchain_documents"
    EmbeddingDimension int     // required; e.g., 1536 for OpenAI
    DistanceMetric     DistanceMetric // Cosine (default), L2, InnerProduct
    Embedder           embeddings.Embedder
    AutoCreateTable    bool // default: true
    CollectionName     string // optional namespace within a table
}

type DistanceMetric int
const (
    Cosine       DistanceMetric = iota // <=>
    L2                                  // <->
    InnerProduct                        // <#>
)
```

#### Store: `vectorstores/pgvector/store.go`

1. **`New(ctx, opts...)`** — create store, optionally run migrations:
   - `CREATE EXTENSION IF NOT EXISTS vector`
   - Create table:
     ```sql
     CREATE TABLE IF NOT EXISTS langchain_documents (
         id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
         content    TEXT NOT NULL,
         metadata   JSONB DEFAULT '{}',
         embedding  vector(1536),
         collection TEXT DEFAULT ''
     );
     CREATE INDEX IF NOT EXISTS idx_embedding ON langchain_documents
         USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
     ```

2. **`AddDocuments(ctx, docs)`**:
   - Compute embeddings via `Embedder.EmbedDocuments()`.
   - Batch INSERT with `pgx` or `database/sql` using prepared statements.
   - Return generated UUIDs.
   - Use `ON CONFLICT (id) DO UPDATE` for upsert when documents have pre-set IDs.

3. **`SimilaritySearch(ctx, query, k)`**:
   - Embed the query via `Embedder.EmbedQuery()`.
   - Execute:
     ```sql
     SELECT id, content, metadata
     FROM langchain_documents
     WHERE collection = $1
     ORDER BY embedding <=> $2
     LIMIT $3
     ```
   - Apply metadata filters as `AND metadata @> $4::jsonb` when provided.

4. **`SimilaritySearchWithScore(ctx, query, k)`** — same as above but include the distance in the SELECT.

5. **`Delete(ctx, ids)`** — `DELETE FROM langchain_documents WHERE id = ANY($1)`.

6. **`Close()`** — close the underlying `*sql.DB` if the store owns it.

#### Search Options: `vectorstores/pgvector/search.go`

```go
type SearchOption func(*searchConfig)

func WithFilter(metadata map[string]any) SearchOption  // JSONB containment
func WithScoreThreshold(min float64) SearchOption       // filter by minimum score
func WithCollection(name string) SearchOption           // namespace filter
```

### SQL Injection Prevention

- All queries use parameterized statements (`$1`, `$2`, ...).
- Table name is validated against `^[a-zA-Z_][a-zA-Z0-9_]*$` at construction time.
- No string interpolation of user input into SQL.

### Testing Strategy

- Unit tests with a mock `*sql.DB` (using `DATA-DOG/go-sqlmock`) for query verification.
- Integration tests (behind `//go:build integration` tag) against a real PostgreSQL + pgvector instance.
- Test all distance metrics produce valid ordering.
- Test metadata filtering with various JSONB queries.
- Test upsert behavior (add same document twice).
- Test concurrent AddDocuments + SimilaritySearch.

### Dependencies

- `github.com/lib/pq` or `github.com/jackc/pgx/v5` — PostgreSQL driver.
- Should be a separate module (`vectorstores/pgvector/go.mod`) to avoid pulling the driver into the core module for users who don't need it.
