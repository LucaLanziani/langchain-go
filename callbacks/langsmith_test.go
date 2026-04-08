package callbacks

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestNewLangSmithHandlerDefaults(t *testing.T) {
	h := NewLangSmithHandler("my-project")
	if h == nil {
		t.Fatal("expected non-nil handler")
	}
}

func TestNewLangSmithHandlerEmptyProject(t *testing.T) {
	h := NewLangSmithHandler("")
	if h == nil {
		t.Fatal("expected non-nil handler")
	}
}

// TestLangSmithHandlerNoAPIKey verifies all methods run without API key (no HTTP calls made).
func TestLangSmithHandlerNoAPIKey(t *testing.T) {
	h := NewLangSmithHandler("test-project")
	// apiKey is "" so no HTTP calls will be made.
	ctx := context.Background()

	h.OnChainStart(ctx, map[string]any{"input": "x"}, "r1", "", map[string]any{"name": "TestChain"})
	h.OnChainEnd(ctx, map[string]any{"output": "y"}, "r1")

	h.OnChainStart(ctx, nil, "r2", "", nil)
	h.OnChainError(ctx, errors.New("fail"), "r2")

	h.OnLLMStart(ctx, []string{"p1"}, "r3", "", nil)
	h.OnLLMEnd(ctx, nil, "r3")

	h.OnLLMStart(ctx, []string{"p1"}, "r4", "", nil)
	h.OnLLMError(ctx, errors.New("llm fail"), "r4")

	h.OnChatModelStart(ctx, nil, "r5", "", map[string]any{"name": "gpt-4"})
	h.OnChainEnd(ctx, nil, "r5")

	h.OnToolStart(ctx, "calc", "1+1", "r6", "")
	h.OnToolEnd(ctx, "2", "r6")

	h.OnToolStart(ctx, "bad_tool", "x", "r7", "")
	h.OnToolError(ctx, errors.New("tool fail"), "r7")

	h.OnRetrieverStart(ctx, "query", "r8", "")
	h.OnRetrieverEnd(ctx, []*core.Document{{PageContent: "doc1"}}, "r8")

	h.OnRetrieverStart(ctx, "bad query", "r9", "")
	h.OnRetrieverError(ctx, errors.New("retriever fail"), "r9")

	// endRun on unknown runID — should not panic.
	h.OnChainEnd(ctx, nil, "unknown-run")
}

// TestLangSmithHandlerWithMockServer tests actual HTTP posting when API key is set.
func TestLangSmithHandlerWithMockServer(t *testing.T) {
	received := make([]string, 0)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received = append(received, r.Method+" "+r.URL.Path)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	h := &LangSmithHandler{
		apiKey:   "test-key",
		endpoint: server.URL,
		project:  "test",
		client:   server.Client(),
		runs:     make(map[string]*langSmithRun),
	}

	ctx := context.Background()
	h.OnChainStart(ctx, map[string]any{}, "run-1", "", map[string]any{"name": "TestChain"})
	h.OnChainEnd(ctx, map[string]any{}, "run-1")
	if err := h.Flush(ctx); err != nil {
		t.Fatalf("Flush error: %v", err)
	}
	if len(received) != 2 {
		t.Fatalf("expected 2 requests after flush, got %d (%v)", len(received), received)
	}
}

func TestLangSmithHandlerCloseStopsNewEvents(t *testing.T) {
	received := make([]string, 0)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		received = append(received, r.Method+" "+r.URL.Path)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	h := &LangSmithHandler{
		apiKey:   "test-key",
		endpoint: server.URL,
		project:  "test",
		client:   server.Client(),
		runs:     make(map[string]*langSmithRun),
	}

	ctx := context.Background()
	h.OnChainStart(ctx, map[string]any{}, "run-1", "", map[string]any{"name": "TestChain"})
	h.OnChainEnd(ctx, map[string]any{}, "run-1")
	if err := h.Close(ctx); err != nil {
		t.Fatalf("Close error: %v", err)
	}

	before := len(received)
	h.OnChainStart(ctx, map[string]any{}, "run-2", "", map[string]any{"name": "Ignored"})
	h.OnChainEnd(ctx, map[string]any{}, "run-2")
	if len(received) != before {
		t.Fatalf("expected closed handler to stop new events, got %v", received)
	}
}
