package memory

import (
	"context"
	"fmt"

	"github.com/LucaLanziani/langchain-go/core"
)

// ConversationBufferMemory stores the entire conversation history.
// It implements the Memory interface.
type ConversationBufferMemory struct {
	// ChatHistory is the backing message store used when no persistent history is set.
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

	// MaxMessages is the maximum number of messages to retain in history.
	// 0 means unlimited. When exceeded, the oldest messages are dropped.
	// Note: MaxMessages is ignored when a persistent history backend is used.
	MaxMessages int

	// persistentHistory is an optional persistent backend. When set, it takes
	// priority over ChatHistory for all read/write operations.
	persistentHistory PersistentHistory
}

// NewConversationBufferMemory creates a new ConversationBufferMemory.
// Pass MemoryOption values (e.g. WithChatHistory) to customize behaviour.
func NewConversationBufferMemory(opts ...MemoryOption) *ConversationBufferMemory {
	cfg := &memoryConfig{}
	for _, o := range opts {
		o(cfg)
	}
	m := &ConversationBufferMemory{
		ChatHistory:    NewChatMessageHistory(),
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
func (m *ConversationBufferMemory) MemoryVariables() []string {
	return []string{m.MemoryKey}
}

// LoadMemoryVariables loads the conversation history.
// If a persistent history backend is configured, Load is called first to
// refresh messages from the backend before returning them.
func (m *ConversationBufferMemory) LoadMemoryVariables(ctx context.Context, _ map[string]any) (map[string]any, error) {
	if m.persistentHistory != nil {
		if err := m.persistentHistory.Load(ctx); err != nil {
			return nil, err
		}
		messages := m.persistentHistory.GetMessages(ctx)
		if m.ReturnMessages {
			return map[string]any{m.MemoryKey: messages}, nil
		}
		return map[string]any{
			m.MemoryKey: core.GetBufferString(messages, m.HumanPrefix, m.AIPrefix),
		}, nil
	}

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
// If a persistent history backend is configured, Save is called after adding
// messages to flush them to the backend.
func (m *ConversationBufferMemory) SaveContext(ctx context.Context, inputs map[string]any, outputs map[string]any) error {
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
	if m.MaxMessages > 0 {
		msgs := m.ChatHistory.GetMessages(ctx)
		if len(msgs) > m.MaxMessages {
			m.ChatHistory.SetMessages(ctx, msgs[len(msgs)-m.MaxMessages:])
		}
	}
	return nil
}

// Clear resets the conversation history.
func (m *ConversationBufferMemory) Clear(ctx context.Context) error {
	if m.persistentHistory != nil {
		m.persistentHistory.Clear(ctx)
		return nil
	}
	m.ChatHistory.Clear(ctx)
	return nil
}

// toString converts a value to its string representation.
func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprint(v)
}

// Ensure ConversationBufferMemory implements Memory.
var _ Memory = (*ConversationBufferMemory)(nil)
