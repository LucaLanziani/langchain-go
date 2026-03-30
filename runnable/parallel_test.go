package runnable

import (
	"context"
	"errors"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

func TestParallel(t *testing.T) {
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	addTen := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i + 10, nil },
		name: "addTen",
	}

	p := NewParallel[int, int](map[string]core.Runnable[int, int]{
		"double": double,
		"addTen": addTen,
	})

	result, err := p.Invoke(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["double"] != 10 {
		t.Errorf("expected double=10, got %v", result["double"])
	}
	if result["addTen"] != 15 {
		t.Errorf("expected addTen=15, got %v", result["addTen"])
	}
}

func TestParallelAny(t *testing.T) {
	p := NewParallelAny[string](map[string]func(ctx context.Context, input string, opts ...core.Option) (any, error){
		"upper": func(_ context.Context, s string, _ ...core.Option) (any, error) {
			result := ""
			for _, c := range s {
				if c >= 'a' && c <= 'z' {
					result += string(rune(c - 32))
				} else {
					result += string(c)
				}
			}
			return result, nil
		},
		"len": func(_ context.Context, s string, _ ...core.Option) (any, error) {
			return len(s), nil
		},
	})

	result, err := p.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["upper"] != "HELLO" {
		t.Errorf("expected upper=HELLO, got %v", result["upper"])
	}
	if result["len"] != 5 {
		t.Errorf("expected len=5, got %v", result["len"])
	}
}

func TestParallelGetName(t *testing.T) {
	p := NewParallel[int, int](map[string]core.Runnable[int, int]{})
	if p.GetName() != "RunnableParallel" {
		t.Errorf("expected 'RunnableParallel', got %q", p.GetName())
	}
	p.WithName("MyParallel")
	if p.GetName() != "MyParallel" {
		t.Errorf("expected 'MyParallel', got %q", p.GetName())
	}
}

func TestParallelStream(t *testing.T) {
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	p := NewParallel[int, int](map[string]core.Runnable[int, int]{"double": double})

	iter, err := p.Stream(context.Background(), 3)
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
	if result["double"] != 6 {
		t.Errorf("expected double=6, got %v", result["double"])
	}
}

func TestParallelBatch(t *testing.T) {
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	p := NewParallel[int, int](map[string]core.Runnable[int, int]{"double": double})

	results, err := p.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0]["double"] != 2 || results[1]["double"] != 4 || results[2]["double"] != 6 {
		t.Errorf("unexpected results: %v", results)
	}
}

func TestParallelError(t *testing.T) {
	failing := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return 0, errors.New("branch error") },
		name: "fail",
	}
	p := NewParallel[int, int](map[string]core.Runnable[int, int]{"fail": failing})

	_, err := p.Invoke(context.Background(), 1)
	if err == nil {
		t.Error("expected error from failing branch")
	}
}

func TestParallelMaxConcurrency(t *testing.T) {
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	p := NewParallel[int, int](map[string]core.Runnable[int, int]{"double": double})

	// WithMaxConcurrency limits concurrency.
	result, err := p.Invoke(context.Background(), 5, core.WithMaxConcurrency(1))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result["double"] != 10 {
		t.Errorf("expected double=10, got %v", result["double"])
	}
}
