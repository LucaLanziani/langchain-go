package memory

import (
	"context"

	"github.com/LucaLanziani/langchain-go/core"
)

// ConversationBufferMemory stores the entire conversation history.
// It implements the Memory interface.
type ConversationBufferMemory struct {
	// ChatHistory is the backing message store.
	ChatHistory *ChatMessageHistory

	// MemoryKey is the key used to store/retrieve messages. Default: "history".
	MemoryKey string

	// InputKey is the key for the human input. Default: "input".
	InputKey string

	// OutputKey is the key for the AI output. Default: "output".
	OutputKey string

	// ReturnMessages controls whether to return messages or a formatted string.
	ReturnMessages bool

	// HumanPrefix is the prefix for human messages in string output.
	HumanPrefix string

	// AIPrefix is the prefix for AI messages in string output.
	AIPrefix string
}

// NewConversationBufferMemory creates a new ConversationBufferMemory.
func NewConversationBufferMemory() *ConversationBufferMemory {
	return &ConversationBufferMemory{
		ChatHistory:    NewChatMessageHistory(),
		MemoryKey:      "history",
		InputKey:       "input",
		OutputKey:      "output",
		ReturnMessages: false,
		HumanPrefix:    "Human",
		AIPrefix:       "AI",
	}
}

// MemoryVariables returns the keys this memory produces.
func (m *ConversationBufferMemory) MemoryVariables() []string {
	return []string{m.MemoryKey}
}

// LoadMemoryVariables loads the conversation history.
func (m *ConversationBufferMemory) LoadMemoryVariables(ctx context.Context, _ map[string]any) (map[string]any, error) {
	messages := m.ChatHistory.GetMessages(ctx)

	if m.ReturnMessages {
		return map[string]any{
			m.MemoryKey: messages,
		}, nil
	}

	return map[string]any{
		m.MemoryKey: core.GetBufferString(messages, m.HumanPrefix, m.AIPrefix),
	}, nil
}

// SaveContext saves the input and output messages.
func (m *ConversationBufferMemory) SaveContext(ctx context.Context, inputs map[string]any, outputs map[string]any) error {
	inputVal, ok := inputs[m.InputKey]
	if ok {
		m.ChatHistory.AddUserMessage(ctx, toString(inputVal))
	}
	outputVal, ok := outputs[m.OutputKey]
	if ok {
		m.ChatHistory.AddAIMessage(ctx, toString(outputVal))
	}
	return nil
}

// Clear resets the conversation history.
func (m *ConversationBufferMemory) Clear(ctx context.Context) error {
	m.ChatHistory.Clear(ctx)
	return nil
}

// toString converts a value to its string representation.
func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

// Ensure ConversationBufferMemory implements Memory.
var _ Memory = (*ConversationBufferMemory)(nil)
