package runnable

import (
	"context"

	"github.com/langchain-go/langchain-go/core"
)

// Lambda wraps a Go function as a Runnable.
// It implements Runnable[I, O].
type Lambda[I, O any] struct {
	fn   func(ctx context.Context, input I) (O, error)
	name string
}

// NewLambda creates a new Lambda runnable from a function.
func NewLambda[I, O any](fn func(ctx context.Context, input I) (O, error)) *Lambda[I, O] {
	return &Lambda[I, O]{fn: fn}
}

// WithName sets the name for tracing.
func (l *Lambda[I, O]) WithName(name string) *Lambda[I, O] {
	l.name = name
	return l
}

// GetName returns the name of this lambda.
func (l *Lambda[I, O]) GetName() string {
	if l.name != "" {
		return l.name
	}
	return "RunnableLambda"
}

// Invoke runs the wrapped function.
func (l *Lambda[I, O]) Invoke(ctx context.Context, input I, opts ...core.Option) (O, error) {
	return l.fn(ctx, input)
}

// Stream returns a single-chunk stream of the function result.
func (l *Lambda[I, O]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[O], error) {
	result, err := l.fn(ctx, input)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[O], 1)
	ch <- core.StreamChunk[O]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the function for each input.
func (l *Lambda[I, O]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]O, error) {
	results := make([]O, len(inputs))
	for i, input := range inputs {
		result, err := l.fn(ctx, input)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}
