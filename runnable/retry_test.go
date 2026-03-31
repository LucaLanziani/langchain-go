package runnable

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/LucaLanziani/langchain-go/core"
)

type retryTrackingCallback struct {
	core.BaseCallbackHandler
	mu     sync.Mutex
	events []core.RetryData
}

func (c *retryTrackingCallback) OnRetry(_ context.Context, data core.RetryData) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = append(c.events, data)
}

func TestRetryInvokeEventuallySucceeds(t *testing.T) {
	var attempts int32
	r := NewLambda(func(_ context.Context, input int) (int, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n <= 2 {
			return 0, errors.New("status 503")
		}
		return input * 2, nil
	})

	cb := &retryTrackingCallback{}
	wrapped := WithRetry[int, int](
		r,
		RetryMaxAttempts(4),
		RetryInitialBackoff(5*time.Millisecond),
		RetryJitter(false),
	)

	result, err := wrapped.Invoke(context.Background(), 5, core.WithCallbacks(cb))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != 10 {
		t.Fatalf("expected 10, got %d", result)
	}
	if got := atomic.LoadInt32(&attempts); got != 3 {
		t.Fatalf("expected 3 attempts, got %d", got)
	}
	if len(cb.events) != 2 {
		t.Fatalf("expected 2 retry events, got %d", len(cb.events))
	}
	if cb.events[0].Attempt != 2 || cb.events[1].Attempt != 3 {
		t.Fatalf("unexpected retry attempts: %+v", cb.events)
	}
}

func TestRetryStreamRetriesOpen(t *testing.T) {
	var attempts int32
	r := &mockRunnable[string, string]{
		name: "streamer",
		fn: func(_ context.Context, input string) (string, error) {
			n := atomic.AddInt32(&attempts, 1)
			if n == 1 {
				return "", errors.New("status 429")
			}
			return input + "!", nil
		},
	}

	wrapped := WithRetry[string, string](
		r,
		RetryMaxAttempts(3),
		RetryInitialBackoff(5*time.Millisecond),
		RetryJitter(false),
	)

	iter, err := wrapped.Stream(context.Background(), "ok")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	chunk, ok, err := iter.Next()
	if err != nil {
		t.Fatalf("unexpected stream error: %v", err)
	}
	if !ok || chunk != "ok!" {
		t.Fatalf("unexpected stream result ok=%v chunk=%q", ok, chunk)
	}
	if got := atomic.LoadInt32(&attempts); got != 2 {
		t.Fatalf("expected 2 attempts, got %d", got)
	}
}

func TestRetryBatchRetriesEachItemIndependently(t *testing.T) {
	attempts := map[int]int{}
	var mu sync.Mutex
	r := NewLambda(func(_ context.Context, input int) (int, error) {
		mu.Lock()
		attempts[input]++
		current := attempts[input]
		mu.Unlock()

		if input == 2 && current == 1 {
			return 0, errors.New("status 500")
		}
		return input * 10, nil
	})

	wrapped := WithRetry[int, int](
		r,
		RetryMaxAttempts(3),
		RetryInitialBackoff(5*time.Millisecond),
		RetryJitter(false),
	)

	results, err := wrapped.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := []int{10, 20, 30}
	for i := range expected {
		if results[i] != expected[i] {
			t.Fatalf("results[%d]=%d expected %d", i, results[i], expected[i])
		}
	}

	mu.Lock()
	defer mu.Unlock()
	if attempts[1] != 1 || attempts[2] != 2 || attempts[3] != 1 {
		t.Fatalf("unexpected attempt counts: %+v", attempts)
	}
}

func TestRetryContextCancellationDuringBackoff(t *testing.T) {
	var attempts int32
	r := NewLambda(func(_ context.Context, _ string) (string, error) {
		atomic.AddInt32(&attempts, 1)
		return "", errors.New("status 503")
	})
	wrapped := WithRetry[string, string](
		r,
		RetryMaxAttempts(5),
		RetryInitialBackoff(200*time.Millisecond),
		RetryJitter(false),
	)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(25 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := wrapped.Invoke(ctx, "x")
	elapsed := time.Since(start)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context canceled, got %v", err)
	}
	if elapsed >= 150*time.Millisecond {
		t.Fatalf("expected early return on cancellation, took %v", elapsed)
	}
	if got := atomic.LoadInt32(&attempts); got != 1 {
		t.Fatalf("expected only one attempt before cancellation, got %d", got)
	}
}

func TestDefaultRetryableError(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		{err: errors.New("OpenAI API error (status 429): rate limited"), want: true},
		{err: fmt.Errorf("wrapped: %w", errors.New("status 503")), want: true},
		{err: context.Canceled, want: false},
		{err: errors.New("validation failed"), want: false},
	}

	for _, tc := range cases {
		got := DefaultRetryableError(tc.err)
		if got != tc.want {
			t.Fatalf("DefaultRetryableError(%v)=%v, want %v", tc.err, got, tc.want)
		}
	}
}
