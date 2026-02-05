package runnable

import (
	"context"
	"testing"
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
