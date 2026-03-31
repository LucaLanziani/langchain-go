package memory

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/LucaLanziani/langchain-go/core"
)

// SQLHistory is a PersistentHistory backend that stores messages in a SQL database.
// It works with any database/sql-compatible driver (PostgreSQL, SQLite, MySQL, etc.).
// Messages are scoped to a session ID so multiple independent conversations can coexist.
// Thread-safe for concurrent access.
//
// The default CREATE TABLE DDL uses SQLite-compatible syntax (INTEGER PRIMARY KEY).
// For PostgreSQL or MySQL, create the table using your migration tooling and then
// pass the table name via WithTableName. A compatible PostgreSQL schema is:
//
//	CREATE TABLE IF NOT EXISTS chat_messages (
//	    id         SERIAL PRIMARY KEY,
//	    session_id TEXT        NOT NULL,
//	    role       TEXT        NOT NULL,
//	    content    TEXT        NOT NULL,
//	    name       TEXT,
//	    tool_calls JSONB,
//	    metadata   JSONB,
//	    created_at TIMESTAMPTZ DEFAULT NOW()
//	);
//	CREATE INDEX IF NOT EXISTS idx_session ON chat_messages (session_id, id);
type SQLHistory struct {
	db        *sql.DB
	tableName string
	sessionID string

	mu       sync.Mutex
	messages []core.Message
	lastErr  error // records the most recent background persistence error
}

// NewSQLHistory creates a SQLHistory and ensures the backing table exists using
// a SQLite-compatible DDL. For other databases, create the table via migrations
// before calling this function.
// The caller is responsible for providing an open *sql.DB with the appropriate
// driver registered.
func NewSQLHistory(db *sql.DB, opts ...HistoryOption) (*SQLHistory, error) {
	cfg := defaultHistoryConfig()
	for _, o := range opts {
		o(cfg)
	}
	h := &SQLHistory{
		db:        db,
		tableName: cfg.tableName,
		sessionID: cfg.sessionID,
	}
	if err := h.createTable(context.Background()); err != nil {
		return nil, err
	}
	return h, nil
}

// createTable runs the DDL to create the messages table if it does not already exist.
// The schema uses INTEGER PRIMARY KEY (auto-incrementing rowid alias in SQLite).
// For other databases, prefer creating the table via dedicated migration tooling.
func (h *SQLHistory) createTable(ctx context.Context) error {
	ddl := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		id         INTEGER PRIMARY KEY,
		session_id TEXT    NOT NULL,
		role       TEXT    NOT NULL,
		content    TEXT    NOT NULL,
		name       TEXT,
		tool_calls TEXT,
		metadata   TEXT,
		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
	)`, h.tableName)
	if _, err := h.db.ExecContext(ctx, ddl); err != nil {
		return fmt.Errorf("sqlhistory: create table: %w", err)
	}
	return nil
}

// AddMessage inserts the message into the database and appends it to the
// in-memory buffer. Persistence errors are stored and accessible via Err().
// The message is always added to the in-memory buffer regardless of any DB error.
func (h *SQLHistory) AddMessage(ctx context.Context, msg core.Message) {
	h.mu.Lock()
	h.messages = append(h.messages, msg)
	h.mu.Unlock()

	sm := messageToStored(msg)
	toolCallsJSON, _ := json.Marshal(sm.ToolCalls)
	metaJSON, _ := json.Marshal(sm.AdditionalKwargs)

	query := fmt.Sprintf(
		`INSERT INTO %s (session_id, role, content, name, tool_calls, metadata) VALUES (?, ?, ?, ?, ?, ?)`,
		h.tableName,
	)
	_, err := h.db.ExecContext(ctx, query,
		h.sessionID,
		sm.Type,
		sm.Content,
		sm.Name,
		string(toolCallsJSON),
		string(metaJSON),
	)
	if err != nil {
		h.mu.Lock()
		h.lastErr = fmt.Errorf("sqlhistory: insert message: %w", err)
		h.mu.Unlock()
	}
}

// GetMessages returns a copy of all in-memory messages.
func (h *SQLHistory) GetMessages(_ context.Context) []core.Message {
	h.mu.Lock()
	defer h.mu.Unlock()
	result := make([]core.Message, len(h.messages))
	copy(result, h.messages)
	return result
}

// Clear deletes all rows for the current session from the database and clears
// the in-memory buffer. Persistence errors are stored and accessible via Err().
func (h *SQLHistory) Clear(ctx context.Context) {
	h.mu.Lock()
	h.messages = nil
	h.mu.Unlock()

	query := fmt.Sprintf(`DELETE FROM %s WHERE session_id = ?`, h.tableName)
	if _, err := h.db.ExecContext(ctx, query, h.sessionID); err != nil {
		h.mu.Lock()
		h.lastErr = fmt.Errorf("sqlhistory: clear session: %w", err)
		h.mu.Unlock()
	}
}

// Err returns the most recent background persistence error (e.g. a failed INSERT
// during AddMessage or a failed DELETE during Clear). It is reset on the next
// successful operation.
func (h *SQLHistory) Err() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.lastErr
}

// Load fetches all messages for the current session from the database and
// replaces the in-memory buffer with them.
func (h *SQLHistory) Load(ctx context.Context) error {
	query := fmt.Sprintf(
		`SELECT role, content, name, tool_calls, metadata FROM %s WHERE session_id = ? ORDER BY id`,
		h.tableName,
	)
	rows, err := h.db.QueryContext(ctx, query, h.sessionID)
	if err != nil {
		return fmt.Errorf("sqlhistory: load: %w", err)
	}
	defer rows.Close()

	var msgs []core.Message
	for rows.Next() {
		var role, content string
		var nameVal, toolCallsRaw, metaRaw sql.NullString
		if err := rows.Scan(&role, &content, &nameVal, &toolCallsRaw, &metaRaw); err != nil {
			return fmt.Errorf("sqlhistory: scan row: %w", err)
		}

		sm := storedMessage{
			Type:    role,
			Content: content,
		}
		if nameVal.Valid {
			sm.Name = nameVal.String
		}
		if toolCallsRaw.Valid && toolCallsRaw.String != "" && toolCallsRaw.String != "null" {
			_ = json.Unmarshal([]byte(toolCallsRaw.String), &sm.ToolCalls)
		}
		if metaRaw.Valid && metaRaw.String != "" && metaRaw.String != "null" {
			_ = json.Unmarshal([]byte(metaRaw.String), &sm.AdditionalKwargs)
		}

		msg, err := storedToMessage(sm)
		if err != nil {
			return fmt.Errorf("sqlhistory: decode message: %w", err)
		}
		msgs = append(msgs, msg)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("sqlhistory: rows: %w", err)
	}

	h.mu.Lock()
	h.messages = msgs
	h.lastErr = nil
	h.mu.Unlock()
	return nil
}

// Save is a no-op for SQLHistory because AddMessage persists each message
// immediately. It exists to satisfy the PersistentHistory interface.
func (h *SQLHistory) Save(_ context.Context) error {
	return nil
}

// ListSessions returns all distinct session IDs present in the table.
func (h *SQLHistory) ListSessions(ctx context.Context) ([]string, error) {
	query := fmt.Sprintf(`SELECT DISTINCT session_id FROM %s ORDER BY session_id`, h.tableName)
	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("sqlhistory: list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("sqlhistory: scan session id: %w", err)
		}
		sessions = append(sessions, id)
	}
	return sessions, rows.Err()
}

// DeleteSession removes all rows for the specified session from the database.
func (h *SQLHistory) DeleteSession(ctx context.Context, sessionID string) error {
	query := fmt.Sprintf(`DELETE FROM %s WHERE session_id = ?`, h.tableName)
	_, err := h.db.ExecContext(ctx, query, sessionID)
	if err != nil {
		return fmt.Errorf("sqlhistory: delete session %q: %w", sessionID, err)
	}
	return nil
}

// Ensure SQLHistory implements PersistentHistory.
var _ PersistentHistory = (*SQLHistory)(nil)

