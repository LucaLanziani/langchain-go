# Feature 006: Persistent Conversation History

> **GitHub Issue:** [#6](https://github.com/LucaLanziani/langchain-go/issues/6)

## User Story

**As a** developer building a chatbot or multi-turn agent,
**I want** conversation history to be persisted across application restarts,
**so that** users can resume conversations seamlessly, and I don't lose context when my service redeploys or scales horizontally.

### Acceptance Criteria

- A `PersistentChatHistory` implements the existing `memory.ChatMessageHistory` interface but backs storage with pluggable backends.
- Built-in backends: file-system (JSON) and SQL (PostgreSQL/SQLite).
- Conversations are keyed by a session ID so multiple independent conversations can coexist.
- The interface supports listing, loading, and deleting conversations by session ID.
- I can use persistent history as a drop-in replacement in `ConversationBufferMemory` and `ConversationWindowMemory`.
- Thread-safe for concurrent access to different sessions (and optionally the same session with locking).
- Messages include timestamps for audit and replay.

### Example Usage

```go
import "github.com/LucaLanziani/langchain-go/memory"

// File-based persistence
history := memory.NewFileHistory("./chat-data",
    memory.WithSessionID("user-123-session-456"),
)

mem := memory.NewConversationBufferMemory(
    memory.WithChatHistory(history),
)

// Use in a chain or agent as usual
chain := chains.NewLLMChain(model, prompt,
    chains.WithMemory(mem),
)

// SQL-based persistence
history := memory.NewSQLHistory(db,
    memory.WithSessionID("session-789"),
    memory.WithTableName("chat_messages"),
)

// List all sessions
sessions, _ := history.ListSessions(ctx)

// Delete a session
_ = history.DeleteSession(ctx, "old-session")
```

---

## Implementation Plan

### Extended Interface: `memory/history.go`

```go
type PersistentHistory interface {
    // Embeds the existing in-memory interface
    AddMessage(msg core.Message)
    GetMessages() []core.Message
    Clear()

    // Persistence operations
    Load(ctx context.Context) error              // load from backend
    Save(ctx context.Context) error              // flush to backend
    ListSessions(ctx context.Context) ([]string, error)
    DeleteSession(ctx context.Context, sessionID string) error
}
```

### Storage Format

Each message stored with:
```json
{
    "type": "human|ai|system|tool",
    "content": "...",
    "name": "...",
    "tool_calls": [...],
    "additional_kwargs": {...},
    "timestamp": "2025-03-31T12:00:00Z"
}
```

### File Backend: `memory/file_history.go`

- One JSON file per session: `{base_dir}/{session_id}.json`.
- `Load`: read and deserialize the file.
- `Save`: serialize and write atomically (temp file + rename).
- `AddMessage`: appends to in-memory buffer + auto-saves (configurable: immediate or batched).
- `ListSessions`: list `*.json` files in the base directory.
- `DeleteSession`: remove the file.
- File locking with `flock` for concurrent process safety.

### SQL Backend: `memory/sql_history.go`

- Table schema:
  ```sql
  CREATE TABLE IF NOT EXISTS chat_messages (
      id         SERIAL PRIMARY KEY,
      session_id TEXT NOT NULL,
      role       TEXT NOT NULL,
      content    TEXT NOT NULL,
      name       TEXT,
      tool_calls JSONB,
      metadata   JSONB,
      created_at TIMESTAMPTZ DEFAULT NOW()
  );
  CREATE INDEX idx_session ON chat_messages(session_id, created_at);
  ```
- `Load`: `SELECT ... WHERE session_id = $1 ORDER BY created_at`.
- `Save / AddMessage`: `INSERT INTO chat_messages ...`.
- `Clear`: `DELETE FROM chat_messages WHERE session_id = $1`.
- `DeleteSession`: same as Clear.
- `ListSessions`: `SELECT DISTINCT session_id FROM chat_messages`.
- Uses `database/sql` interface — works with any SQL driver.

### Integration with Existing Memory

Modify `ConversationBufferMemory` and `ConversationWindowMemory`:
- Add `WithChatHistory(PersistentHistory)` option.
- On `LoadMemoryVariables`: call `history.Load(ctx)` first if history implements `PersistentHistory`.
- On `SaveContext`: call `history.Save(ctx)` after adding messages.

### Testing Strategy

- Unit tests for file backend with temp directories.
- Unit tests for SQL backend with go-sqlmock.
- Test message round-trip: add messages → save → load → verify identical.
- Test multi-session isolation.
- Test concurrent access to different sessions.
- Test atomic file writes (crash mid-write shouldn't corrupt).
- Integration test with real SQLite (stdlib, no external deps).

### Dependencies

- File backend: no new dependencies (stdlib `os`, `encoding/json`).
- SQL backend: uses `database/sql` interface — driver is user's choice.
