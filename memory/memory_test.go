package memory

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestChatMessageHistory(t *testing.T) {
	ctx := context.Background()
	h := NewChatMessageHistory()

	h.AddUserMessage(ctx, "Hello")
	h.AddAIMessage(ctx, "Hi there!")

	messages := h.GetMessages(ctx)
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	if messages[0].GetType() != core.MessageTypeHuman {
		t.Errorf("expected human message, got %s", messages[0].GetType())
	}
	if messages[1].GetType() != core.MessageTypeAI {
		t.Errorf("expected AI message, got %s", messages[1].GetType())
	}

	h.Clear(ctx)
	messages = h.GetMessages(ctx)
	if len(messages) != 0 {
		t.Errorf("expected 0 messages after clear, got %d", len(messages))
	}
}

func TestConversationBufferMemory(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationBufferMemory()

	// Save context.
	err := mem.SaveContext(ctx,
		map[string]any{"input": "Hello"},
		map[string]any{"output": "Hi!"},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Load memory variables (string format).
	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	history, ok := vars["history"].(string)
	if !ok {
		t.Fatal("expected string history")
	}
	if history == "" {
		t.Error("expected non-empty history")
	}

	// Test memory variables key.
	keys := mem.MemoryVariables()
	if len(keys) != 1 || keys[0] != "history" {
		t.Errorf("expected [history], got %v", keys)
	}
}

func TestConversationBufferMemoryReturnMessages(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationBufferMemory()
	mem.ReturnMessages = true

	err := mem.SaveContext(ctx,
		map[string]any{"input": "Hello"},
		map[string]any{"output": "Hi!"},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	messages, ok := vars["history"].([]core.Message)
	if !ok {
		t.Fatal("expected []core.Message history")
	}
	if len(messages) != 2 {
		t.Errorf("expected 2 messages, got %d", len(messages))
	}
}

func TestConversationWindowMemory(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationWindowMemory(2) // Keep last 2 turns.
	mem.ReturnMessages = true

	// Add 3 turns.
	for i := 0; i < 3; i++ {
		err := mem.SaveContext(ctx,
			map[string]any{"input": "q"},
			map[string]any{"output": "a"},
		)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	messages, ok := vars["history"].([]core.Message)
	if !ok {
		t.Fatal("expected []core.Message history")
	}
	// 2 turns * 2 messages = 4 messages.
	if len(messages) != 4 {
		t.Errorf("expected 4 messages (2 turns), got %d", len(messages))
	}
}

func TestConversationWindowMemoryClear(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationWindowMemory(5)

	err := mem.SaveContext(ctx,
		map[string]any{"input": "test"},
		map[string]any{"output": "response"},
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = mem.Clear(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	mem.ReturnMessages = true
	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	messages := vars["history"].([]core.Message)
	if len(messages) != 0 {
		t.Errorf("expected 0 messages after clear, got %d", len(messages))
	}
}
