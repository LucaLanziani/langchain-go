package callbacks

import (
	"context"
	"errors"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

// trackingHandler records which methods were called.
type trackingHandler struct {
	core.BaseCallbackHandler
	called []string
}

func (h *trackingHandler) OnLLMStart(_ context.Context, _ []string, _, _ string, _ map[string]any) {
	h.called = append(h.called, "OnLLMStart")
}
func (h *trackingHandler) OnChatModelStart(_ context.Context, _ []core.Message, _, _ string, _ map[string]any) {
	h.called = append(h.called, "OnChatModelStart")
}
func (h *trackingHandler) OnLLMNewToken(_ context.Context, _ string, _ string) {
	h.called = append(h.called, "OnLLMNewToken")
}
func (h *trackingHandler) OnLLMEnd(_ context.Context, _ *core.LLMResult, _ string) {
	h.called = append(h.called, "OnLLMEnd")
}
func (h *trackingHandler) OnLLMError(_ context.Context, _ error, _ string) {
	h.called = append(h.called, "OnLLMError")
}
func (h *trackingHandler) OnChainStart(_ context.Context, _ map[string]any, _, _ string, _ map[string]any) {
	h.called = append(h.called, "OnChainStart")
}
func (h *trackingHandler) OnChainEnd(_ context.Context, _ map[string]any, _ string) {
	h.called = append(h.called, "OnChainEnd")
}
func (h *trackingHandler) OnChainError(_ context.Context, _ error, _ string) {
	h.called = append(h.called, "OnChainError")
}
func (h *trackingHandler) OnToolStart(_ context.Context, _, _ string, _, _ string) {
	h.called = append(h.called, "OnToolStart")
}
func (h *trackingHandler) OnToolEnd(_ context.Context, _ string, _ string) {
	h.called = append(h.called, "OnToolEnd")
}
func (h *trackingHandler) OnToolError(_ context.Context, _ error, _ string) {
	h.called = append(h.called, "OnToolError")
}
func (h *trackingHandler) OnAgentAction(_ context.Context, _ core.AgentActionData, _ string) {
	h.called = append(h.called, "OnAgentAction")
}
func (h *trackingHandler) OnAgentFinish(_ context.Context, _ core.AgentFinishData, _ string) {
	h.called = append(h.called, "OnAgentFinish")
}
func (h *trackingHandler) OnRetrieverStart(_ context.Context, _ string, _, _ string) {
	h.called = append(h.called, "OnRetrieverStart")
}
func (h *trackingHandler) OnRetrieverEnd(_ context.Context, _ []*core.Document, _ string) {
	h.called = append(h.called, "OnRetrieverEnd")
}
func (h *trackingHandler) OnRetrieverError(_ context.Context, _ error, _ string) {
	h.called = append(h.called, "OnRetrieverError")
}
func (h *trackingHandler) OnText(_ context.Context, _ string, _ string) {
	h.called = append(h.called, "OnText")
}
func (h *trackingHandler) OnRetry(_ context.Context, _ core.RetryData) {
	h.called = append(h.called, "OnRetry")
}

func TestManagerDispatchesAll(t *testing.T) {
	h := &trackingHandler{}
	m := NewManager(h)
	ctx := context.Background()

	m.OnLLMStart(ctx, []string{"prompt"}, "run1", "", nil)
	m.OnChatModelStart(ctx, nil, "run1", "", nil)
	m.OnLLMNewToken(ctx, "tok", "run1")
	m.OnLLMEnd(ctx, nil, "run1")
	m.OnLLMError(ctx, errors.New("e"), "run1")
	m.OnChainStart(ctx, nil, "run1", "", nil)
	m.OnChainEnd(ctx, nil, "run1")
	m.OnChainError(ctx, errors.New("e"), "run1")
	m.OnToolStart(ctx, "tool", "input", "run1", "")
	m.OnToolEnd(ctx, "output", "run1")
	m.OnToolError(ctx, errors.New("e"), "run1")
	m.OnAgentAction(ctx, core.AgentActionData{}, "run1")
	m.OnAgentFinish(ctx, core.AgentFinishData{}, "run1")
	m.OnRetrieverStart(ctx, "query", "run1", "")
	m.OnRetrieverEnd(ctx, nil, "run1")
	m.OnRetrieverError(ctx, errors.New("e"), "run1")
	m.OnText(ctx, "text", "run1")
	m.OnRetry(ctx, core.RetryData{})

	expected := []string{
		"OnLLMStart", "OnChatModelStart", "OnLLMNewToken", "OnLLMEnd", "OnLLMError",
		"OnChainStart", "OnChainEnd", "OnChainError",
		"OnToolStart", "OnToolEnd", "OnToolError",
		"OnAgentAction", "OnAgentFinish",
		"OnRetrieverStart", "OnRetrieverEnd", "OnRetrieverError",
		"OnText", "OnRetry",
	}
	if len(h.called) != len(expected) {
		t.Errorf("expected %d calls, got %d: %v", len(expected), len(h.called), h.called)
		return
	}
	for i, e := range expected {
		if h.called[i] != e {
			t.Errorf("call[%d]: expected %q, got %q", i, e, h.called[i])
		}
	}
}

func TestManagerMultipleHandlers(t *testing.T) {
	h1 := &trackingHandler{}
	h2 := &trackingHandler{}
	m := NewManager(h1, h2)
	m.OnLLMStart(context.Background(), []string{"p"}, "r", "", nil)
	if len(h1.called) != 1 || len(h2.called) != 1 {
		t.Error("expected both handlers to receive the event")
	}
}

func TestManagerWithInheritableHandlers(t *testing.T) {
	h := &trackingHandler{}
	m := NewManager()
	m.WithInheritableHandlers(h)

	child := m.GetChild("step1")
	child.OnLLMStart(context.Background(), nil, "r", "", nil)
	if len(h.called) != 1 {
		t.Error("expected inheritable handler in child to receive event")
	}
}

func TestManagerGetChild(t *testing.T) {
	m := NewManager()
	child := m.GetChild("mytag")
	if child == nil {
		t.Fatal("expected non-nil child manager")
	}
}

func TestManagerWithOptions(t *testing.T) {
	h := &trackingHandler{}
	m := NewManager(h)
	m.WithParentRunID("parent-1")
	m.WithTags("tag1", "tag2")
	m.WithMetadata(map[string]any{"key": "val"})
	// Just verify chaining doesn't panic.
	m.AllHandlers()
}
