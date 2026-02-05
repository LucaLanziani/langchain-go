// Package runnable provides composition primitives for building chains
// from Runnable components (the Go equivalent of LCEL).
package runnable

import (
	"context"
	"fmt"

	"github.com/LucaLanziani/langchain-go/core"
)

// step wraps a single runnable in the sequence with type erasure,
// allowing heterogeneous runnables to be composed.
type step struct {
	name   string
	invoke func(ctx context.Context, input any, opts ...core.Option) (any, error)
}

// Sequence chains multiple runnables together: the output of each becomes the input of the next.
// Because Go generics don't support heterogeneous type lists, the intermediate
// types are erased to `any`. The overall Sequence is typed on the first input
// and last output.
type Sequence[I, O any] struct {
	steps []step
	name  string
}

// GetName returns the name of the sequence.
func (s *Sequence[I, O]) GetName() string {
	if s.name != "" {
		return s.name
	}
	return "RunnableSequence"
}

// WithName sets the sequence name for tracing.
func (s *Sequence[I, O]) WithName(name string) *Sequence[I, O] {
	s.name = name
	return s
}

// Invoke runs all steps sequentially, passing each output as the next input.
func (s *Sequence[I, O]) Invoke(ctx context.Context, input I, opts ...core.Option) (O, error) {
	var current any = input
	var zero O
	for i, st := range s.steps {
		result, err := st.invoke(ctx, current, opts...)
		if err != nil {
			return zero, fmt.Errorf("step %d (%s): %w", i, st.name, err)
		}
		current = result
	}
	output, ok := current.(O)
	if !ok {
		return zero, fmt.Errorf("final step output type mismatch: got %T, want %T", current, zero)
	}
	return output, nil
}

// Stream runs all steps sequentially and returns the output of the last step as a stream.
// Currently this is a simple implementation that invokes all steps and streams the final result.
func (s *Sequence[I, O]) Stream(ctx context.Context, input I, opts ...core.Option) (*core.StreamIterator[O], error) {
	result, err := s.Invoke(ctx, input, opts...)
	if err != nil {
		return nil, err
	}
	ch := make(chan core.StreamChunk[O], 1)
	ch <- core.StreamChunk[O]{Value: result}
	close(ch)
	return core.NewStreamIterator(ch), nil
}

// Batch runs the sequence for multiple inputs.
func (s *Sequence[I, O]) Batch(ctx context.Context, inputs []I, opts ...core.Option) ([]O, error) {
	results := make([]O, len(inputs))
	for i, input := range inputs {
		result, err := s.Invoke(ctx, input, opts...)
		if err != nil {
			return nil, fmt.Errorf("batch item %d: %w", i, err)
		}
		results[i] = result
	}
	return results, nil
}

// Pipe2 chains two runnables into a Sequence.
func Pipe2[A, B, C any](
	first core.Runnable[A, B],
	second core.Runnable[B, C],
) *Sequence[A, C] {
	return &Sequence[A, C]{
		steps: []step{
			{name: first.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return first.Invoke(ctx, input.(A), opts...)
			}},
			{name: second.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return second.Invoke(ctx, input.(B), opts...)
			}},
		},
	}
}

// Pipe3 chains three runnables into a Sequence.
func Pipe3[A, B, C, D any](
	first core.Runnable[A, B],
	second core.Runnable[B, C],
	third core.Runnable[C, D],
) *Sequence[A, D] {
	return &Sequence[A, D]{
		steps: []step{
			{name: first.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return first.Invoke(ctx, input.(A), opts...)
			}},
			{name: second.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return second.Invoke(ctx, input.(B), opts...)
			}},
			{name: third.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return third.Invoke(ctx, input.(C), opts...)
			}},
		},
	}
}

// Pipe4 chains four runnables into a Sequence.
func Pipe4[A, B, C, D, E any](
	first core.Runnable[A, B],
	second core.Runnable[B, C],
	third core.Runnable[C, D],
	fourth core.Runnable[D, E],
) *Sequence[A, E] {
	return &Sequence[A, E]{
		steps: []step{
			{name: first.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return first.Invoke(ctx, input.(A), opts...)
			}},
			{name: second.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return second.Invoke(ctx, input.(B), opts...)
			}},
			{name: third.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return third.Invoke(ctx, input.(C), opts...)
			}},
			{name: fourth.GetName(), invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return fourth.Invoke(ctx, input.(D), opts...)
			}},
		},
	}
}

// Pipe creates a type-erased sequence from an untyped series of steps.
// Each step must accept and produce `any`. This is the most flexible
// but least type-safe variant.
func Pipe(runnables ...core.Runnable[any, any]) *Sequence[any, any] {
	steps := make([]step, len(runnables))
	for i, r := range runnables {
		r := r // capture
		steps[i] = step{
			name: r.GetName(),
			invoke: func(ctx context.Context, input any, opts ...core.Option) (any, error) {
				return r.Invoke(ctx, input, opts...)
			},
		}
	}
	return &Sequence[any, any]{steps: steps}
}
