package callbacks

import (
	"context"
	"errors"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestNewStdoutHandler(t *testing.T) {
	h := NewStdoutHandler()
	if h == nil {
		t.Fatal("expected non-nil handler")
	}
	if !h.Color {
		t.Error("expected Color to be true by default")
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"hello", 10, "hello"},
		{"hello world", 5, "hello..."},
		{"", 5, ""},
		{"abcdef", 3, "abc..."},
	}

	for _, tt := range tests {
		result := truncate(tt.input, tt.maxLen)
		if result != tt.expected {
			t.Errorf("truncate(%q, %d): expected %q, got %q", tt.input, tt.maxLen, tt.expected, result)
		}
	}
}

func TestStdoutHandlerMethods(t *testing.T) {
	h := NewStdoutHandler()
	h.Color = false // suppress color codes in test output
	ctx := context.Background()

	// Just verify none of these panic.
	h.OnChainStart(ctx, map[string]any{}, "r", "", map[string]any{"name": "TestChain"})
	h.OnChainEnd(ctx, nil, "r")
	h.OnChainError(ctx, errors.New("error"), "r")
	h.OnLLMStart(ctx, []string{"prompt1", "prompt2"}, "r", "", nil)
	h.OnChatModelStart(ctx, []core.Message{core.NewHumanMessage("hi")}, "r", "", nil)
	h.OnLLMNewToken(ctx, "token", "r")
	h.OnLLMEnd(ctx, nil, "r")
	h.OnLLMError(ctx, errors.New("llm err"), "r")
	h.OnToolStart(ctx, "tool", "input", "r", "")
	h.OnToolEnd(ctx, "output", "r")
	h.OnToolError(ctx, errors.New("tool err"), "r")
	h.OnAgentAction(ctx, core.AgentActionData{Tool: "search", ToolInput: "query"}, "r")
	h.OnAgentFinish(ctx, core.AgentFinishData{Output: map[string]any{"result": "done"}}, "r")
	h.OnRetrieverStart(ctx, "query", "r", "")
	h.OnRetrieverEnd(ctx, []*core.Document{{PageContent: "doc1"}}, "r")
	h.OnText(ctx, "some text", "r")
}

func TestStdoutHandlerWithColor(t *testing.T) {
	h := NewStdoutHandler()
	h.Color = true
	ctx := context.Background()
	// Just verify it doesn't panic.
	h.OnChainStart(ctx, nil, "r", "", nil)
}
