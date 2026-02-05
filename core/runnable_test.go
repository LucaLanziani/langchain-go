package core

import (
	"testing"
)

func TestStreamIterator(t *testing.T) {
	ch := make(chan StreamChunk[string], 3)
	ch <- StreamChunk[string]{Value: "a"}
	ch <- StreamChunk[string]{Value: "b"}
	ch <- StreamChunk[string]{Value: "c"}
	close(ch)

	iter := NewStreamIterator(ch)

	// Read all values.
	val, ok, err := iter.Next()
	if err != nil || !ok || val != "a" {
		t.Errorf("expected (a, true, nil), got (%q, %v, %v)", val, ok, err)
	}

	val, ok, err = iter.Next()
	if err != nil || !ok || val != "b" {
		t.Errorf("expected (b, true, nil), got (%q, %v, %v)", val, ok, err)
	}

	val, ok, err = iter.Next()
	if err != nil || !ok || val != "c" {
		t.Errorf("expected (c, true, nil), got (%q, %v, %v)", val, ok, err)
	}

	// Stream is exhausted.
	_, ok, err = iter.Next()
	if err != nil || ok {
		t.Errorf("expected (_, false, nil), got (_, %v, %v)", ok, err)
	}
}

func TestStreamIteratorCollect(t *testing.T) {
	ch := make(chan StreamChunk[int], 3)
	ch <- StreamChunk[int]{Value: 1}
	ch <- StreamChunk[int]{Value: 2}
	ch <- StreamChunk[int]{Value: 3}
	close(ch)

	iter := NewStreamIterator(ch)
	vals, err := iter.Collect()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vals) != 3 || vals[0] != 1 || vals[1] != 2 || vals[2] != 3 {
		t.Errorf("expected [1,2,3], got %v", vals)
	}
}

func TestStreamIteratorClose(t *testing.T) {
	ch := make(chan StreamChunk[string], 10)
	for i := 0; i < 10; i++ {
		ch <- StreamChunk[string]{Value: "x"}
	}
	close(ch)

	iter := NewStreamIterator(ch)
	iter.Close()

	// After close, Next should return false.
	_, ok, err := iter.Next()
	if err != nil || ok {
		t.Errorf("expected (_, false, nil) after close, got (_, %v, %v)", ok, err)
	}
}

func TestStreamIteratorError(t *testing.T) {
	ch := make(chan StreamChunk[string], 2)
	ch <- StreamChunk[string]{Value: "ok"}
	ch <- StreamChunk[string]{Err: ErrTest}
	close(ch)

	iter := NewStreamIterator(ch)
	val, ok, err := iter.Next()
	if err != nil || !ok || val != "ok" {
		t.Errorf("first Next: expected (ok, true, nil), got (%q, %v, %v)", val, ok, err)
	}

	_, _, err = iter.Next()
	if err != ErrTest {
		t.Errorf("second Next: expected ErrTest, got %v", err)
	}
}

var ErrTest = &testError{}

type testError struct{}

func (e *testError) Error() string { return "test error" }
