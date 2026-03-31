package memory

import (
	"context"

	"github.com/LucaLanziani/langchain-go/core"
)

// PersistentHistory is the interface for pluggable persistent chat history backends.
// Implementations back message storage with durable backends (file, SQL, etc.) while
// also keeping messages in memory for fast access within a session.
type PersistentHistory interface {
	// AddMessage appends a message to the history and persists it.
	AddMessage(ctx context.Context, msg core.Message)

	// GetMessages returns all messages for the current session.
	GetMessages(ctx context.Context) []core.Message

	// Clear removes all in-memory messages and persists the cleared state.
	Clear(ctx context.Context)

	// Load reads messages from the backend into the in-memory buffer.
	Load(ctx context.Context) error

	// Save flushes in-memory messages to the backend.
	Save(ctx context.Context) error

	// ListSessions returns all session IDs known to the backend.
	ListSessions(ctx context.Context) ([]string, error)

	// DeleteSession permanently removes all messages for the given session from the backend.
	DeleteSession(ctx context.Context, sessionID string) error
}

// HistoryOption configures a persistent history instance.
type HistoryOption func(*historyConfig)

type historyConfig struct {
	sessionID string
	tableName string
	autoSave  bool
}

func defaultHistoryConfig() *historyConfig {
	return &historyConfig{
		sessionID: "default",
		tableName: "chat_messages",
		autoSave:  true,
	}
}

// WithSessionID sets the session ID used to scope messages in the backend.
func WithSessionID(id string) HistoryOption {
	return func(c *historyConfig) { c.sessionID = id }
}

// WithTableName sets the SQL table name (SQL backend only).
func WithTableName(name string) HistoryOption {
	return func(c *historyConfig) { c.tableName = name }
}

// WithAutoSave controls whether AddMessage automatically persists to the backend.
// Default is true. Set to false to batch writes and call Save manually.
func WithAutoSave(enabled bool) HistoryOption {
	return func(c *historyConfig) { c.autoSave = enabled }
}

// MemoryOption configures ConversationBufferMemory or ConversationWindowMemory.
type MemoryOption func(*memoryConfig)

type memoryConfig struct {
	chatHistory PersistentHistory
}

// WithChatHistory sets a persistent chat history backend for conversation memory.
// When set, LoadMemoryVariables calls Load before returning messages, and
// SaveContext calls Save after adding messages.
func WithChatHistory(h PersistentHistory) MemoryOption {
	return func(c *memoryConfig) { c.chatHistory = h }
}
