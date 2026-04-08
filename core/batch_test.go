package core

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestBatchRunsInParallelByDefault(t *testing.T) {
	inputs := []int{1, 2, 3, 4}
	var inFlight int32
	var maxInFlight int32

	results, err := Batch(context.Background(), inputs, nil, func(ctx context.Context, input int, _ ...Option) (int, error) {
		current := atomic.AddInt32(&inFlight, 1)
		for {
			max := atomic.LoadInt32(&maxInFlight)
			if current <= max || atomic.CompareAndSwapInt32(&maxInFlight, max, current) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
		return input * 2, nil
	})
	if err != nil {
		t.Fatalf("Batch error: %v", err)
	}
	if len(results) != len(inputs) {
		t.Fatalf("expected %d results, got %d", len(inputs), len(results))
	}
	if maxInFlight < 2 {
		t.Fatalf("expected parallel execution, max concurrency was %d", maxInFlight)
	}
}

func TestBatchHonorsMaxConcurrency(t *testing.T) {
	inputs := []int{1, 2, 3, 4, 5}
	var inFlight int32
	var maxInFlight int32

	_, err := Batch(context.Background(), inputs, []Option{WithMaxConcurrency(2)}, func(ctx context.Context, input int, _ ...Option) (int, error) {
		current := atomic.AddInt32(&inFlight, 1)
		for {
			max := atomic.LoadInt32(&maxInFlight)
			if current <= max || atomic.CompareAndSwapInt32(&maxInFlight, max, current) {
				break
			}
		}
		time.Sleep(20 * time.Millisecond)
		atomic.AddInt32(&inFlight, -1)
		return input, nil
	})
	if err != nil {
		t.Fatalf("Batch error: %v", err)
	}
	if maxInFlight > 2 {
		t.Fatalf("expected max concurrency <= 2, got %d", maxInFlight)
	}
}
