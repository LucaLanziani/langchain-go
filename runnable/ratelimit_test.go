package runnable

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

func TestRateLimitInvokeDelaysRequests(t *testing.T) {
	r := NewLambda(func(_ context.Context, input int) (int, error) { return input, nil })
	limited := WithRateLimit[int, int](r, RateLimitRPS(20), RateLimitBurst(1))

	_, err := limited.Invoke(context.Background(), 1)
	if err != nil {
		t.Fatalf("first invoke failed: %v", err)
	}

	start := time.Now()
	_, err = limited.Invoke(context.Background(), 2)
	if err != nil {
		t.Fatalf("second invoke failed: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed < 40*time.Millisecond {
		t.Fatalf("expected rate limited delay, got %v", elapsed)
	}
}

func TestRateLimitRespectsContextCancellation(t *testing.T) {
	r := NewLambda(func(_ context.Context, input int) (int, error) { return input, nil })
	limited := WithRateLimit[int, int](r, RateLimitRPS(2), RateLimitBurst(1))

	_, err := limited.Invoke(context.Background(), 1)
	if err != nil {
		t.Fatalf("first invoke failed: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = limited.Invoke(ctx, 2)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline exceeded, got %v", err)
	}
}

func TestRateLimitBatchAppliesPerItem(t *testing.T) {
	r := NewLambda(func(_ context.Context, input int) (int, error) { return input * 2, nil })
	limited := WithRateLimit[int, int](r, RateLimitRPS(20), RateLimitBurst(1))

	start := time.Now()
	results, err := limited.Batch(context.Background(), []int{1, 2, 3})
	if err != nil {
		t.Fatalf("batch failed: %v", err)
	}
	if len(results) != 3 || results[0] != 2 || results[1] != 4 || results[2] != 6 {
		t.Fatalf("unexpected batch results: %v", results)
	}
	elapsed := time.Since(start)
	if elapsed < 90*time.Millisecond {
		t.Fatalf("expected batch to be rate limited, got %v", elapsed)
	}
}

func TestRetryAndRateLimitCompose(t *testing.T) {
	var attempts int32
	r := NewLambda(func(_ context.Context, input string) (string, error) {
		n := atomic.AddInt32(&attempts, 1)
		if n == 1 {
			return "", errors.New("status 503")
		}
		return input + " ok", nil
	})

	resilient := WithRetry[string, string](
		r,
		RetryMaxAttempts(3),
		RetryInitialBackoff(5*time.Millisecond),
		RetryJitter(false),
	)
	limited := WithRateLimit[string, string](resilient, RateLimitRPS(100), RateLimitBurst(1))

	out, err := limited.Invoke(context.Background(), "hello")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "hello ok" {
		t.Fatalf("unexpected output: %q", out)
	}
	if got := atomic.LoadInt32(&attempts); got != 2 {
		t.Fatalf("expected 2 attempts, got %d", got)
	}
}
