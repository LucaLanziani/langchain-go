package runnable

import (
	"context"
	"testing"

	"github.com/langchain-go/langchain-go/core"
)

func TestBranch(t *testing.T) {
	positive := &mockRunnable[int, string]{
		fn:   func(_ context.Context, i int) (string, error) { return "positive", nil },
		name: "positive",
	}
	negative := &mockRunnable[int, string]{
		fn:   func(_ context.Context, i int) (string, error) { return "negative", nil },
		name: "negative",
	}
	zero := &mockRunnable[int, string]{
		fn:   func(_ context.Context, i int) (string, error) { return "zero", nil },
		name: "zero",
	}

	branch := NewBranch(
		[]BranchCondition[int, string]{
			{Condition: func(i int) bool { return i > 0 }, Runnable: positive},
			{Condition: func(i int) bool { return i < 0 }, Runnable: negative},
		},
		zero,
	)

	tests := []struct {
		input    int
		expected string
	}{
		{5, "positive"},
		{-3, "negative"},
		{0, "zero"},
	}

	for _, tt := range tests {
		result, err := branch.Invoke(context.Background(), tt.input)
		if err != nil {
			t.Fatalf("input %d: unexpected error: %v", tt.input, err)
		}
		if result != tt.expected {
			t.Errorf("input %d: expected %q, got %q", tt.input, tt.expected, result)
		}
	}
}

func TestBranchNoDefault(t *testing.T) {
	branch := NewBranch[int, string](
		[]BranchCondition[int, string]{
			{Condition: func(i int) bool { return i > 100 }, Runnable: &mockRunnable[int, string]{
				fn: func(_ context.Context, _ int) (string, error) { return "big", nil },
			}},
		},
		nil,
	)

	_, err := branch.Invoke(context.Background(), 5)
	if err == nil {
		t.Error("expected error when no branch matches and no default")
	}
}

// Need the core import to avoid unused import error.
var _ core.Option = nil
