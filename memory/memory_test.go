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

func TestConversationBufferMemoryClear(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationBufferMemory()
	_ = mem.SaveContext(ctx, map[string]any{"input": "hi"}, map[string]any{"output": "hello"})
	if err := mem.Clear(ctx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	vars, _ := mem.LoadMemoryVariables(ctx, nil)
	if vars["history"].(string) != "" {
		t.Error("expected empty history after clear")
	}
}

func TestConversationBufferMemoryMaxMessages(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationBufferMemory()
	mem.MaxMessages = 4 // Keep only last 4 messages (2 turns).

	for i := 0; i < 4; i++ {
		_ = mem.SaveContext(ctx, map[string]any{"input": "q"}, map[string]any{"output": "a"})
	}

	mem.ReturnMessages = true
	vars, _ := mem.LoadMemoryVariables(ctx, nil)
	msgs := vars["history"].([]core.Message)
	if len(msgs) != 4 {
		t.Errorf("expected 4 messages (limited), got %d", len(msgs))
	}
}

func TestConversationBufferMemoryNonStringInput(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationBufferMemory()
	// Non-string values — should not blow up.
	_ = mem.SaveContext(ctx, map[string]any{"input": 123}, map[string]any{"output": 456})
	vars, _ := mem.LoadMemoryVariables(ctx, nil)
	// toString() returns "" for non-string, so history string should not crash.
	_ = vars
}

func TestChatMessageHistoryAddMessage(t *testing.T) {
	ctx := context.Background()
	h := NewChatMessageHistory()
	h.AddMessage(ctx, core.NewHumanMessage("msg via AddMessage"))
	msgs := h.GetMessages(ctx)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
}

func TestChatMessageHistorySetMessages(t *testing.T) {
	ctx := context.Background()
	h := NewChatMessageHistory()
	h.AddUserMessage(ctx, "original")
	h.SetMessages(ctx, []core.Message{core.NewAIMessage("replaced")})
	msgs := h.GetMessages(ctx)
	if len(msgs) != 1 || msgs[0].GetType() != core.MessageTypeAI {
		t.Errorf("expected replaced AI message, got %v", msgs)
	}
}

func TestConversationWindowMemoryStringOutput(t *testing.T) {
	ctx := context.Background()
	mem := NewConversationWindowMemory(2)
	mem.ReturnMessages = false
	_ = mem.SaveContext(ctx, map[string]any{"input": "hello"}, map[string]any{"output": "hi"})

	vars, err := mem.LoadMemoryVariables(ctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	hist, ok := vars["history"].(string)
	if !ok || hist == "" {
		t.Errorf("expected non-empty string history, got %T", vars["history"])
	}
}

func TestConversationWindowMemoryVariables(t *testing.T) {
	mem := NewConversationWindowMemory(5)
	keys := mem.MemoryVariables()
	if len(keys) != 1 || keys[0] != "history" {
		t.Errorf("expected [history], got %v", keys)
	}
}
