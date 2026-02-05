package runnable

import (
	"context"

	"github.com/langchain-go/langchain-go/core"
)

// Passthrough passes its input through unchanged.
// It implements Runnable[T, T].
type Passthrough[T any] struct {
	name string
}

// NewPassthrough creates a new Passthrough runnable.
func NewPassthrough[T any]() *Passthrough[T] {
	return &Passthrough[T]{}
}

// WithName sets the name for tracing.
func (p *Passthrough[T]) WithName(name string) *Passthrough[T] {
	p.name = name
	return p
}

// GetName returns the name of this passthrough.
func (p *Passthrough[T]) GetName() string {
	if p.name != "" {
		return p.name
	}
	return "RunnablePassthrough"
}

// Invoke returns the input unchanged.
func (p *Passthrough[T]) Invoke(ctx context.Context, input T, opts ...core.Option) (T, error) {
	return input, nil
}

// Stream returns a single-chunk stream of the input.
func (p *Passthrough[T]) Stream(ctx context.Context, input T, opts ...core.Option) (*core.StreamIterator[T], error) {
	ch := make(chan core.StreamChunk[T], 1)
	ch <- core.StreamChunk[T]{Value: input}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch returns the inputs unchanged.
func (p *Passthrough[T]) Batch(ctx context.Context, inputs []T, opts ...core.Option) ([]T, error) {
	result := make([]T, len(inputs))
	copy(result, inputs)
	return result, nil
}

// Assign creates a parallel runnable that passes the input through unchanged
// while also computing additional keys. The result is a map[string]any containing
// the original input plus the computed values.
//
// Usage:
//
//	chain := Assign(
//	    map[string]core.Runnable[map[string]any, any]{
//	        "context": retriever,
//	    },
//	)
//
// This adds a "context" key to the input map while preserving existing keys.
type Assign[I any] struct {
	additions map[string]func(ctx context.Context, input I, opts ...core.Option) (any, error)
	keys      []string
	name      string
}

// NewAssign creates a new Assign runnable.
func NewAssign[I any](additions map[string]func(ctx context.Context, input I, opts ...core.Option) (any, error)) *Assign[I] {
	keys := make([]string, 0, len(additions))
	for k := range additions {
		keys = append(keys, k)
	}
	return &Assign[I]{additions: additions, keys: keys}
}

// GetName returns the name.
func (a *Assign[I]) GetName() string {
	if a.name != "" {
		return a.name
	}
	return "RunnableAssign"
}

// Invoke passes through the input and adds computed keys.
func (a *Assign[I]) Invoke(ctx context.Context, input I, opts ...core.Option) (map[string]any, error) {
	result := make(map[string]any)
	// If input is a map, copy its values.
	if m, ok := any(input).(map[string]any); ok {
		for k, v := range m {
			result[k] = v
		}
	}
	// Compute additional keys.
	for _, key := range a.keys {
		val, err := a.additions[key](ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		result[key] = val
	}
	return result, nil
}

// Stream returns a single-chunk stream.
func (a *Assign[I]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[map[string]any], error) {
	result, err := a.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[map[string]any], 1)
	ch <- core.StreamChunk[map[string]any]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs for multiple inputs.
func (a *Assign[I]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]map[string]any, error) {
	results := make([]map[string]any, len(inputs))
	for i, input := range inputs {
		result, err := a.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}
