package memory

import (
	"context"

	"github.com/LucaLanziani/langchain-go/core"
)

// ConversationWindowMemory stores a sliding window of the most recent K
// conversation turns. It implements the Memory interface.
type ConversationWindowMemory struct {
	// ChatHistory is the backing message store used when no persistent history is set.
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

	// persistentHistory is an optional persistent backend. When set, it takes
	// priority over ChatHistory for all read/write operations.
	persistentHistory PersistentHistory
}

// NewConversationWindowMemory creates a new ConversationWindowMemory with K turns.
// Pass MemoryOption values (e.g. WithChatHistory) to customize behaviour.
func NewConversationWindowMemory(k int, opts ...MemoryOption) *ConversationWindowMemory {
	cfg := &memoryConfig{}
	for _, o := range opts {
		o(cfg)
	}
	m := &ConversationWindowMemory{
		ChatHistory:    NewChatMessageHistory(),
		K:              k,
		MemoryKey:      "history",
		InputKey:       "input",
		OutputKey:      "output",
		ReturnMessages: false,
		HumanPrefix:    "Human",
		AIPrefix:       "AI",
	}
	if cfg.chatHistory != nil {
		m.persistentHistory = cfg.chatHistory
	}
	return m
}

// MemoryVariables returns the keys this memory produces.
func (m *ConversationWindowMemory) MemoryVariables() []string {
	return []string{m.MemoryKey}
}

// LoadMemoryVariables loads the last K turns of conversation.
// If a persistent history backend is configured, Load is called first to
// refresh messages from the backend before applying the window.
func (m *ConversationWindowMemory) LoadMemoryVariables(ctx context.Context, _ map[string]any) (map[string]any, error) {
	var messages []core.Message

	if m.persistentHistory != nil {
		if err := m.persistentHistory.Load(ctx); err != nil {
			return nil, err
		}
		messages = m.persistentHistory.GetMessages(ctx)
	} else {
		messages = m.ChatHistory.GetMessages(ctx)
	}

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
// If a persistent history backend is configured, Save is called after adding
// messages to flush them to the backend.
func (m *ConversationWindowMemory) SaveContext(ctx context.Context, inputs map[string]any, outputs map[string]any) error {
	if m.persistentHistory != nil {
		if inputVal, ok := inputs[m.InputKey]; ok {
			m.persistentHistory.AddMessage(ctx, core.NewHumanMessage(toString(inputVal)))
		}
		if outputVal, ok := outputs[m.OutputKey]; ok {
			m.persistentHistory.AddMessage(ctx, core.NewAIMessage(toString(outputVal)))
		}
		return m.persistentHistory.Save(ctx)
	}

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
	if m.persistentHistory != nil {
		m.persistentHistory.Clear(ctx)
		return nil
	}
	m.ChatHistory.Clear(ctx)
	return nil
}

// Ensure ConversationWindowMemory implements Memory.
var _ Memory = (*ConversationWindowMemory)(nil)
