package core

import (
	"context"
	"sync"
)

// Runnable is the core interface that all LangChain components implement.
// It provides a uniform interface for invoking, streaming, and batching
// operations across prompts, models, parsers, retrievers, and tools.
//
// In Go, we don't split into sync/async variants. Use goroutines and
// context.Context for concurrency control.
type Runnable[I, O any] interface {
	// Invoke transforms a single input into an output.
	Invoke(ctx context.Context, input I, opts ...Option) (O, error)

	// Stream transforms an input and streams output chunks as they are produced.
	// The caller must consume the returned StreamIterator until it is exhausted
	// or call Close() to release resources.
	Stream(ctx context.Context, input I, opts ...Option) (*StreamIterator[O], error)

	// Batch transforms multiple inputs in parallel.
	// Implementations honor WithMaxConcurrency when they delegate to the shared
	// batch helper used throughout this library.
	Batch(ctx context.Context, inputs []I, opts ...Option) ([]O, error)

	// GetName returns the name of this runnable for tracing and debugging.
	GetName() string
}

// StreamIterator provides a pull-based iterator for streaming results.
// It wraps a channel internally but exposes a simpler API.
type StreamIterator[T any] struct {
	ch     <-chan StreamChunk[T]
	done   chan struct{}
	closed bool
	mu     sync.Mutex
}

// StreamChunk wraps a streaming value with an optional error.
type StreamChunk[T any] struct {
	Value T
	Err   error
}

// NewStreamIterator creates a new StreamIterator from a channel.
// The producer should close the channel when done.
func NewStreamIterator[T any](ch <-chan StreamChunk[T]) *StreamIterator[T] {
	return &StreamIterator[T]{
		ch:   ch,
		done: make(chan struct{}),
	}
}

// Next returns the next chunk from the stream.
// Returns false when the stream is exhausted.
func (s *StreamIterator[T]) Next() (T, bool, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		var zero T
		return zero, false, nil
	}
	s.mu.Unlock()

	select {
	case <-s.done:
		var zero T
		return zero, false, nil
	case chunk, ok := <-s.ch:
		if !ok {
			var zero T
			return zero, false, nil
		}
		if chunk.Err != nil {
			var zero T
			return zero, false, chunk.Err
		}
		return chunk.Value, true, nil
	}
}

// Collect reads all remaining chunks and returns them as a slice.
func (s *StreamIterator[T]) Collect() ([]T, error) {
	var results []T
	for {
		val, ok, err := s.Next()
		if err != nil {
			return results, err
		}
		if !ok {
			break
		}
		results = append(results, val)
	}
	return results, nil
}

// Close signals the stream is no longer needed.
func (s *StreamIterator[T]) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.closed = true
		close(s.done)
		// Drain remaining items from the channel to unblock the producer.
		go func() {
			for range s.ch {
			}
		}()
	}
}

// Done returns a channel that is closed when the iterator is closed.
func (s *StreamIterator[T]) Done() <-chan struct{} {
	return s.done
}
