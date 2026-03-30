package runnable

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestLambda(t *testing.T) {
	upper := NewLambda(func(_ context.Context, s string) (string, error) {
		return strings.ToUpper(s), nil
	})

	result, err := upper.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != "HELLO" {
		t.Errorf("expected 'HELLO', got %q", result)
	}
}

func TestLambdaBatch(t *testing.T) {
	double := NewLambda(func(_ context.Context, n int) (int, error) {
		return n * 2, nil
	})

	results, err := double.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 3 || results[0] != 2 || results[1] != 4 || results[2] != 6 {
		t.Errorf("expected [2,4,6], got %v", results)
	}
}

func TestLambdaGetName(t *testing.T) {
	l := NewLambda(func(_ context.Context, s string) (string, error) { return s, nil })
	if l.GetName() != "RunnableLambda" {
		t.Errorf("expected 'RunnableLambda', got %q", l.GetName())
	}
	l.WithName("MyLambda")
	if l.GetName() != "MyLambda" {
		t.Errorf("expected 'MyLambda', got %q", l.GetName())
	}
}

func TestLambdaStream(t *testing.T) {
	upper := NewLambda(func(_ context.Context, s string) (string, error) {
		return strings.ToUpper(s), nil
	})
	iter, err := upper.Stream(context.Background(), "hello")
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
	if result != "HELLO" {
		t.Errorf("expected 'HELLO', got %q", result)
	}
}

func TestLambdaInvokeError(t *testing.T) {
	failing := NewLambda(func(_ context.Context, _ string) (string, error) {
		return "", errors.New("lambda error")
	})
	_, err := failing.Invoke(context.Background(), "input")
	if err == nil || err.Error() != "lambda error" {
		t.Errorf("expected 'lambda error', got %v", err)
	}
}
