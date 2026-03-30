package runnable

import (
	"context"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestPassthrough(t *testing.T) {
	p := NewPassthrough[string]()
	result, err := p.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "hello" {
		t.Errorf("expected 'hello', got %q", result)
	}
}

func TestPassthroughBatch(t *testing.T) {
	p := NewPassthrough[int]()
	results, err := p.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 3 || results[0] != 1 || results[1] != 2 || results[2] != 3 {
		t.Errorf("expected [1,2,3], got %v", results)
	}
}

func TestPassthroughGetName(t *testing.T) {
	p := NewPassthrough[string]()
	if p.GetName() != "RunnablePassthrough" {
		t.Errorf("expected 'RunnablePassthrough', got %q", p.GetName())
	}
}

func TestPassthroughWithName(t *testing.T) {
	p := NewPassthrough[string]().WithName("input")
	if p.GetName() != "input" {
		t.Errorf("expected 'input', got %q", p.GetName())
	}
}

func TestPassthroughStream(t *testing.T) {
	p := NewPassthrough[string]()
	iter, err := p.Stream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	chunk, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if chunk != "hello" {
		t.Errorf("expected 'hello', got %q", chunk)
	}
}

func TestAssign(t *testing.T) {
	assign := NewAssign[map[string]any](map[string]func(ctx context.Context, input map[string]any, opts ...core.Option) (any, error){
		"doubled": func(_ context.Context, m map[string]any, _ ...core.Option) (any, error) {
			n := m["n"].(int)
			return n * 2, nil
		},
	})

	result, err := assign.Invoke(context.Background(), map[string]any{"n": 5})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["n"] != 5 {
		t.Errorf("expected n=5, got %v", result["n"])
	}
	if result["doubled"] != 10 {
		t.Errorf("expected doubled=10, got %v", result["doubled"])
	}
}

func TestAssignGetName(t *testing.T) {
	assign := NewAssign[string](map[string]func(ctx context.Context, input string, opts ...core.Option) (any, error){})
	if assign.GetName() != "RunnableAssign" {
		t.Errorf("expected 'RunnableAssign', got %q", assign.GetName())
	}
}

func TestAssignStream(t *testing.T) {
	assign := NewAssign[map[string]any](map[string]func(ctx context.Context, input map[string]any, opts ...core.Option) (any, error){
		"x": func(_ context.Context, _ map[string]any, _ ...core.Option) (any, error) {
			return 42, nil
		},
	})

	iter, err := assign.Stream(context.Background(), map[string]any{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	result, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected a chunk")
	}
	if result["x"] != 42 {
		t.Errorf("expected x=42, got %v", result["x"])
	}
}

func TestAssignBatch(t *testing.T) {
	assign := NewAssign[map[string]any](map[string]func(ctx context.Context, input map[string]any, opts ...core.Option) (any, error){
		"flag": func(_ context.Context, _ map[string]any, _ ...core.Option) (any, error) {
			return true, nil
		},
	})

	results, err := assign.Batch(context.Background(), []map[string]any{{}, {}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 || results[0]["flag"] != true {
		t.Errorf("unexpected results: %v", results)
	}
}
