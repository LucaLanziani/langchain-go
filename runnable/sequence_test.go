package runnable

import (
	"context"
	"fmt"
	"testing"

	"github.com/LucaLanziani/langchain-go/core"
)

// mockRunnable is a test helper.
type mockRunnable[I, O any] struct {
	fn   func(ctx context.Context, input I) (O, error)
	name string
}

func (m *mockRunnable[I, O]) GetName() string { return m.name }
func (m *mockRunnable[I, O]) Invoke(ctx context.Context, input I, opts ...core.Option) (O, error) {
	return m.fn(ctx, input)
}
func (m *mockRunnable[I, O]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[O], error) {
	result, err := m.fn(ctx, input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[O], 1)
	ch <- core.StreamChunk[O]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}
func (m *mockRunnable[I, O]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]O, error) {
	results := make([]O, len(inputs))
	for i, in := range inputs {
		r, err := m.fn(ctx, in)
		if err != nil {
			return nil, err
		}
		results[i] = r
	}
	return results, nil
}

func TestPipe2(t *testing.T) {
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	toString := &mockRunnable[int, string]{
		fn:   func(_ context.Context, i int) (string, error) { return fmt.Sprintf("%d", i), nil },
		name: "toString",
	}

	chain := Pipe2(double, toString)
	result, err := chain.Invoke(context.Background(), 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "10" {
		t.Errorf("expected '10', got %q", result)
	}
}

func TestPipe3(t *testing.T) {
	add1 := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i + 1, nil },
		name: "add1",
	}
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	toString := &mockRunnable[int, string]{
		fn:   func(_ context.Context, i int) (string, error) { return fmt.Sprintf("%d", i), nil },
		name: "toString",
	}

	chain := Pipe3(add1, double, toString)
	result, err := chain.Invoke(context.Background(), 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// (3 + 1) * 2 = 8
	if result != "8" {
		t.Errorf("expected '8', got %q", result)
	}
}

func TestSequenceBatch(t *testing.T) {
	double := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i * 2, nil },
		name: "double",
	}
	add10 := &mockRunnable[int, int]{
		fn:   func(_ context.Context, i int) (int, error) { return i + 10, nil },
		name: "add10",
	}

	chain := Pipe2(double, add10)
	results, err := chain.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := []int{12, 14, 16}
	for i, exp := range expected {
		if results[i] != exp {
			t.Errorf("batch[%d]: expected %d, got %d", i, exp, results[i])
		}
	}
}

func TestSequenceGetName(t *testing.T) {
	r := &mockRunnable[int, int]{fn: func(_ context.Context, i int) (int, error) { return i, nil }, name: "test"}
	chain := Pipe2(r, r)
	if chain.GetName() != "RunnableSequence" {
		t.Errorf("expected 'RunnableSequence', got %q", chain.GetName())
	}
	chain.WithName("MyChain")
	if chain.GetName() != "MyChain" {
		t.Errorf("expected 'MyChain', got %q", chain.GetName())
	}
}
