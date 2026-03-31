package memory

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/LucaLanziani/langchain-go/core"
)

// FileHistory is a PersistentHistory backend that stores messages in JSON files.
// One file per session: {baseDir}/{sessionID}.json.
// Writes are atomic (temp file + rename) to prevent corruption on crash.
// Thread-safe for concurrent access.
type FileHistory struct {
	baseDir   string
	sessionID string
	autoSave  bool

	mu       sync.Mutex
	messages []core.Message
}

// NewFileHistory creates a FileHistory that persists messages under baseDir.
// The directory is created if it does not exist.
func NewFileHistory(baseDir string, opts ...HistoryOption) (*FileHistory, error) {
	cfg := defaultHistoryConfig()
	for _, o := range opts {
		o(cfg)
	}
	if err := os.MkdirAll(baseDir, 0o755); err != nil {
		return nil, fmt.Errorf("filehistory: create base dir: %w", err)
	}
	return &FileHistory{
		baseDir:   baseDir,
		sessionID: cfg.sessionID,
		autoSave:  cfg.autoSave,
	}, nil
}

// sessionFile returns the file path for the current session.
func (h *FileHistory) sessionFile() string {
	return filepath.Join(h.baseDir, h.sessionID+".json")
}

// AddMessage appends a message to the in-memory buffer and, if auto-save is
// enabled, persists the updated history to disk.
func (h *FileHistory) AddMessage(ctx context.Context, msg core.Message) {
	h.mu.Lock()
	h.messages = append(h.messages, msg)
	h.mu.Unlock()

	if h.autoSave {
		// Best-effort save; caller can call Save explicitly to handle errors.
		_ = h.Save(ctx)
	}
}

// GetMessages returns a copy of all in-memory messages.
func (h *FileHistory) GetMessages(_ context.Context) []core.Message {
	h.mu.Lock()
	defer h.mu.Unlock()
	result := make([]core.Message, len(h.messages))
	copy(result, h.messages)
	return result
}

// Clear removes all in-memory messages and overwrites the session file with an
// empty list.
func (h *FileHistory) Clear(ctx context.Context) {
	h.mu.Lock()
	h.messages = nil
	h.mu.Unlock()
	_ = h.Save(ctx)
}

// Load reads the session file and replaces the in-memory buffer with its
// contents. If the file does not exist, the buffer is cleared (no error).
func (h *FileHistory) Load(_ context.Context) error {
	data, err := os.ReadFile(h.sessionFile())
	if os.IsNotExist(err) {
		h.mu.Lock()
		h.messages = nil
		h.mu.Unlock()
		return nil
	}
	if err != nil {
		return fmt.Errorf("filehistory: read session file: %w", err)
	}
	msgs, err := unmarshalMessages(data)
	if err != nil {
		return fmt.Errorf("filehistory: unmarshal messages: %w", err)
	}
	h.mu.Lock()
	h.messages = msgs
	h.mu.Unlock()
	return nil
}

// Save atomically writes the in-memory messages to the session file using a
// temp file + rename to avoid partial writes on crash.
func (h *FileHistory) Save(_ context.Context) error {
	h.mu.Lock()
	data, err := marshalMessages(h.messages)
	h.mu.Unlock()
	if err != nil {
		return fmt.Errorf("filehistory: marshal messages: %w", err)
	}

	// Write to a temp file in the same directory so rename is atomic.
	tmp, err := os.CreateTemp(h.baseDir, ".tmp-"+h.sessionID+"-")
	if err != nil {
		return fmt.Errorf("filehistory: create temp file: %w", err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("filehistory: write temp file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("filehistory: close temp file: %w", err)
	}
	if err := os.Rename(tmpName, h.sessionFile()); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("filehistory: rename temp file: %w", err)
	}
	return nil
}

// ListSessions returns the session IDs of all *.json files in the base directory.
func (h *FileHistory) ListSessions(_ context.Context) ([]string, error) {
	entries, err := os.ReadDir(h.baseDir)
	if err != nil {
		return nil, fmt.Errorf("filehistory: read dir: %w", err)
	}
	var sessions []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasSuffix(name, ".json") {
			sessions = append(sessions, strings.TrimSuffix(name, ".json"))
		}
	}
	return sessions, nil
}

// DeleteSession removes the JSON file for the given session ID.
func (h *FileHistory) DeleteSession(_ context.Context, sessionID string) error {
	path := filepath.Join(h.baseDir, sessionID+".json")
	err := os.Remove(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("filehistory: delete session %q: %w", sessionID, err)
	}
	return nil
}

// Ensure FileHistory implements PersistentHistory.
var _ PersistentHistory = (*FileHistory)(nil)
