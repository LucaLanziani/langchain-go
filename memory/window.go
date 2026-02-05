package memory

import (
	"context"

	"github.com/LucaLanziani/langchain-go/core"
)

// ConversationWindowMemory stores a sliding window of the most recent K
// conversation turns. It implements the Memory interface.
type ConversationWindowMemory struct {
	// ChatHistory is the backing message store.
	ChatHistory *ChatMessageHistory

	// K is the number of recent conversation turns (pairs of messages) to keep.
	K int

	// MemoryKey is the key used to store/retrieve messages. Default: "history".
	MemoryKey string

	// InputKey is the key for the human input.
	InputKey string

	// OutputKey is the key for the AI output.
	OutputKey string

	// ReturnMessages controls whether to return messages or a formatted string.
	ReturnMessages bool

	// HumanPrefix is the prefix for human messages in string output.
	HumanPrefix string

	// AIPrefix is the prefix for AI messages in string output.
	AIPrefix string
}

// NewConversationWindowMemory creates a new ConversationWindowMemory with K turns.
func NewConversationWindowMemory(k int) *ConversationWindowMemory {
	return &ConversationWindowMemory{
		ChatHistory:    NewChatMessageHistory(),
		K:              k,
		MemoryKey:      "history",
		InputKey:       "input",
		OutputKey:      "output",
		ReturnMessages: false,
		HumanPrefix:    "Human",
		AIPrefix:       "AI",
	}
}

// MemoryVariables returns the keys this memory produces.
func (m *ConversationWindowMemory) MemoryVariables() []string {
	return []string{m.MemoryKey}
}

// LoadMemoryVariables loads the last K turns of conversation.
func (m *ConversationWindowMemory) LoadMemoryVariables(ctx context.Context, _ map[string]any) (map[string]any, error) {
	messages := m.ChatHistory.GetMessages(ctx)

	// Keep the last K*2 messages (each turn = 1 human + 1 AI message).
	windowSize := m.K * 2
	if len(messages) > windowSize {
		messages = messages[len(messages)-windowSize:]
	}

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
func (m *ConversationWindowMemory) SaveContext(ctx context.Context, inputs map[string]any, outputs map[string]any) error {
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
func (m *ConversationWindowMemory) Clear(ctx context.Context) error {
	m.ChatHistory.Clear(ctx)
	return nil
}

// Ensure ConversationWindowMemory implements Memory.
var _ Memory = (*ConversationWindowMemory)(nil)
