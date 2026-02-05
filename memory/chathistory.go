package memory

import (
	"context"
	"sync"

	"github.com/LucaLanziani/langchain-go/core"
)

// ChatMessageHistory stores chat messages in memory.
// It is the backing store used by conversation memory implementations.
type ChatMessageHistory struct {
	messages []core.Message
	mu       sync.RWMutex
}

// NewChatMessageHistory creates a new in-memory chat message history.
func NewChatMessageHistory() *ChatMessageHistory {
	return &ChatMessageHistory{}
}

// AddMessage adds a message to the history.
func (h *ChatMessageHistory) AddMessage(_ context.Context, msg core.Message) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = append(h.messages, msg)
}

// AddUserMessage adds a human message.
func (h *ChatMessageHistory) AddUserMessage(_ context.Context, content string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = append(h.messages, core.NewHumanMessage(content))
}

// AddAIMessage adds an AI message.
func (h *ChatMessageHistory) AddAIMessage(_ context.Context, content string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = append(h.messages, core.NewAIMessage(content))
}

// GetMessages returns all messages in the history.
func (h *ChatMessageHistory) GetMessages(_ context.Context) []core.Message {
	h.mu.RLock()
	defer h.mu.RUnlock()
	result := make([]core.Message, len(h.messages))
	copy(result, h.messages)
	return result
}

// Clear removes all messages from the history.
func (h *ChatMessageHistory) Clear(_ context.Context) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.messages = nil
}
