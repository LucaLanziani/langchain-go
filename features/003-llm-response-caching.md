# Feature 003: LLM Response Caching

## User Story

**As a** developer iterating on prompts or running recurring queries,
**I want** to cache LLM responses so that identical inputs return instantly from cache,
**so that** I reduce API costs, speed up development cycles, and avoid unnecessary latency for repeated calls.

### Acceptance Criteria

- I can wrap any `ChatModel` with a caching layer using a one-liner.
- The cache key is derived from: model name, messages content, tool definitions, and generation parameters (temperature, max tokens, etc.).
- Built-in cache backends: in-memory (with TTL + max-size eviction) and file-system.
- Cache interface is pluggable so users can implement Redis, SQLite, or other backends.
- Cache hits bypass the LLM entirely and return the stored `*AIMessage`.
- `Stream` calls on cache hits emit the full response as a single chunk (or optionally replay token-by-token).
- Cache can be disabled per-call with a context option (`core.WithNoCache`).
- Cache entries have configurable TTL (time-to-live).
- Cache statistics (hits, misses, evictions) are exposed through callbacks.

### Example Usage

```go
model := openai.New()

// In-memory cache with 1-hour TTL and 1000 max entries
cached := cache.NewCachedModel(model,
    cache.WithBackend(cache.NewInMemoryBackend(
        cache.MemoryMaxEntries(1000),
        cache.MemoryTTL(1 * time.Hour),
    )),
)

// Uses cache transparently
result, _ := cached.Invoke(ctx, messages) // cache miss → calls LLM
result, _ = cached.Invoke(ctx, messages)  // cache hit → instant

// Bypass cache for a specific call
result, _ = cached.Invoke(ctx, messages, core.WithNoCache())

// File-system cache (persists across runs)
cached := cache.NewCachedModel(model,
    cache.WithBackend(cache.NewFileBackend("/tmp/llm-cache")),
)
```

---

## Implementation Plan

### New Package: `cache/`

#### Interface: `cache/cache.go`

```go
type CacheBackend interface {
    Get(ctx context.Context, key string) (*core.AIMessage, bool, error)
    Set(ctx context.Context, key string, value *core.AIMessage, ttl time.Duration) error
    Clear(ctx context.Context) error
}
```

#### Cache Key Generation: `cache/key.go`

- Serialize the input deterministically:
  - Sort messages, include role + content + tool calls.
  - Include model name, temperature, max tokens, stop sequences, tool definitions.
- Hash with SHA-256 → hex string key.
- Use `encoding/json` for deterministic serialization (sorted map keys).

#### In-Memory Backend: `cache/memory.go`

- `InMemoryBackend` — thread-safe (sync.RWMutex) map with:
  - LRU eviction when `MaxEntries` is exceeded.
  - TTL expiry checked on `Get` (lazy eviction) + periodic cleanup goroutine.
  - Use `container/list` for LRU ordering.

#### File-System Backend: `cache/file.go`

- `FileBackend` — stores each entry as a JSON file in a directory.
  - File name = cache key hash.
  - Each file contains: `{response: AIMessage, expires_at: timestamp}`.
  - `Get` checks file existence and TTL.
  - `Set` writes atomically (write to temp file, then rename).
  - `Clear` removes all files in the cache directory.

#### Cached Model Wrapper: `cache/model.go`

- `CachedModel` — implements `llms.ChatModel`:
  - `Invoke`: compute key → check cache → on miss, call inner model and store result.
  - `Stream`: on cache hit, wrap result in a single-chunk StreamIterator. On miss, delegate to inner model's Stream, collect the full response, cache it, and replay.
  - `Batch`: check cache per item, call inner model only for misses.
  - `Generate`, `BindTools`, `WithStructuredOutput`: delegate to inner model.

#### Context Option

Add `core.WithNoCache()` option that sets a flag in `RunnableConfig.Configurable["no_cache"] = true`. The `CachedModel` checks this flag and skips cache when set.

### Testing Strategy

- Unit tests for key generation: same inputs → same key, different inputs → different keys.
- Unit tests for in-memory backend: set/get, TTL expiry, LRU eviction.
- Unit tests for file backend: set/get, TTL expiry, atomic writes, Clear.
- Integration test: CachedModel with mock ChatModel, verify second call doesn't hit the model.
- Test `WithNoCache` bypasses cache.
- Test thread safety with concurrent goroutines.

### Dependencies

- No external dependencies. Uses stdlib: `crypto/sha256`, `encoding/json`, `container/list`, `os`, `sync`.
